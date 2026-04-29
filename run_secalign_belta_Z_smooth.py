from trl import DPOTrainer, KTOTrainer, ORPOTrainer
import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import transformers
from transformers import EarlyStoppingCallback
from datasets import Dataset


from train import ModelArguments, DataArguments, AttackArguments, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType


@dataclass
class ExperimentArguments:
    train_subset_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of available training samples to use for quick experiments, e.g., 0.1 means 10%"}
    )
    split_seed: int = field(
        default=42,
        metadata={"help": "Fixed random seed for splits to ensure reproducibility"}
    )
    min_val_samples: int = field(
        default=128,
        metadata={"help": "Minimum number of validation samples"}
    )
    min_train_samples: int = field(
        default=256,
        metadata={"help": "Minimum number of training samples"}
    )
    early_stopping_patience: int = field(
        default=2,
        metadata={"help": "Number of eval steps without improvement before stopping"}
    )
    early_stopping_threshold: float = field(
        default=1e-4,
        metadata={"help": "Improvement threshold; smaller than this is not counted"}
    )
    preference_path: str = field(
        default="",
        metadata={"help": "Path to preference dataset JSON file"}
    )
    hard_path: str = field(
        default="",
        metadata={"help": "Path to preference dataset JSON file"}
    )


import torch
import torch.distributed as dist
import torch.nn.functional as F
from trl import DPOTrainer

import random
import copy




# beta-DPO Z-score norm+exp-tanh
class BetaDPOTrainer(DPOTrainer):
    """
    Minimal beta-DPO trainer:
    only implements batch-level dynamic beta from the beta-DPO paper.

    Dynamic rule:
        M_i = beta0 * [ (log pi(yw)-log pref(yw)) - (log pi(yl)-log pref(yl)) ]
        beta_batch = beta0 * (1 + alpha * (mean(M_i) - M0))
        M0 <- momentum * M0 + (1 - momentum) * mean(M_i)
    """

    def __init__(
        self,
        *args,
        beta0: float = 0.1,
        dynamic_alpha: float = 1,
        ema_momentum: float = 0.9,
        beta_min: float = 1e-2,
        beta_max: float = 1.0,
        sync_ddp_stats: bool = True,
        warmup_steps: int = 0, 
        lambda_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.beta0 = float(beta0)
        self.dynamic_alpha = float(dynamic_alpha)
        self.ema_momentum = float(ema_momentum)
        self.beta_min = float(beta_min)
        self.beta_max = beta_max
        self.sync_ddp_stats = sync_ddp_stats
        self.warmup_steps=warmup_steps
        self.lambda_scale = lambda_scale

        # Running EMA baseline M0.
        # Use tensor buffer style so it stays on the right device later.
        self.register_buffer("_beta_dpo_m0", torch.tensor(0.0), persistent=False)
        self.register_buffer("_beta_dpo_initialized", torch.tensor(False), persistent=False)
        self.register_buffer("_beta_dpo_var", torch.tensor(1.0), persistent=False)
        self.std_momentum = float(ema_momentum)
        self.eps = 1e-8

        # Optional: keep last beta for logging/debug.
        self.register_buffer("_last_batch_beta", torch.tensor(self.beta0), persistent=False)

        # logging/debug signals
        self.register_buffer("_last_margin", torch.tensor(0.0), persistent=False)
        self.register_buffer("_last_batch_mean_m", torch.tensor(0.0), persistent=False)
        self.register_buffer("_last_m0", torch.tensor(0.0), persistent=False)

    def register_buffer(self, name, tensor, persistent=False):
        """
        Make this work even though Trainer is not nn.Module.
        We simply attach tensors as attributes.
        """
        setattr(self, name, tensor)

    def _distributed_mean(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute global mean across DDP ranks if requested.
        Expects a scalar tensor.
        """
        if (
            self.sync_ddp_stats
            and dist.is_available()
            and dist.is_initialized()
        ):
            value = value.detach().clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= dist.get_world_size()
        return value

    def _get_dynamic_beta(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Compute batch-level dynamic beta.
        """
        device = policy_chosen_logps.device


        # ===== compute M_i (shared for both warmup & normal) =====
        m_i = self.beta0 * (
            (policy_chosen_logps - reference_chosen_logps)
            - (policy_rejected_logps - reference_rejected_logps)
        )

        batch_mean_m = m_i.detach().mean()
        batch_mean_m = self._distributed_mean(batch_mean_m).to(device)

        # ===== initialize M0 if needed =====
        if not bool(self._beta_dpo_initialized):
            self._beta_dpo_m0 = batch_mean_m.detach()
            self._beta_dpo_initialized = torch.tensor(True, device=device)

        current_m0 = self._beta_dpo_m0.to(device)

        current_var = self._beta_dpo_var.to(device)

        delta = batch_mean_m - current_m0
        batch_var = delta * delta

        new_var = (
            self.std_momentum * current_var
            + (1.0 - self.std_momentum) * batch_var
        )

        self._beta_dpo_var = new_var.detach()
        std = torch.sqrt(new_var + self.eps)


        # ===== warmup: use fixed beta =====
        if self.state.global_step <= self.warmup_steps:

            new_m0 = self.ema_momentum * current_m0 + (1.0 - self.ema_momentum) * batch_mean_m
            self._beta_dpo_m0 = new_m0.detach()
            beta = torch.tensor(self.beta0, device=device)

            # logging
            self._last_batch_beta = beta
            self._last_margin = (batch_mean_m - current_m0).detach()
            self._last_batch_mean_m = batch_mean_m.detach()
            self._last_m0 = current_m0.detach()

            return beta
        


        # reverse beta_movement
        #beta_batch = self.beta0 * (
        #    1.0 - self.dynamic_alpha * (batch_mean_m - current_m0)
        #)

        z = (batch_mean_m - current_m0) / std

        bounded = torch.tanh(z)

        beta_batch = self.beta0 * torch.exp(self.lambda_scale * bounded)

        #beta_batch = self.beta0 * (
        #    1.0 + self.dynamic_alpha * z
        #)

        beta_batch = torch.clamp(beta_batch, min=self.beta_min)
        if self.beta_max is not None:
            beta_batch = torch.clamp(beta_batch, max=self.beta_max)

        # Update EMA baseline after beta is computed.
        # ===== logging signals =====
        delta = batch_mean_m - current_m0
        self._last_batch_beta = beta_batch.detach()
        #self._last_margin = delta.detach()
        self._last_margin = z.detach() 
        self._last_batch_mean_m = batch_mean_m.detach()
        self._last_m0 = current_m0.detach()

        # Update EMA baseline after beta is computed.
        new_m0 = self.ema_momentum * current_m0 + (1.0 - self.ema_momentum) * batch_mean_m
        self._beta_dpo_m0 = new_m0.detach()

        return beta_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Override only the loss. Everything else stays the same as DPOTrainer.
        """
        # Standard DPO log-ratio
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if getattr(self, "reference_free", False):
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        beta_batch = self._get_dynamic_beta(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
        )

        losses = -F.logsigmoid(beta_batch.detach() * logits)

        chosen_rewards = self.beta0 * (
            policy_chosen_logps - reference_chosen_logps
        ).detach()
        rejected_rewards = self.beta0 * (
            policy_rejected_logps - reference_rejected_logps
        ).detach()



        return losses, chosen_rewards, rejected_rewards

    def log(self, logs: dict):
        
        # ===== eval_loss_ori =====
        #if hasattr(self, "_eval_loss_ori_sum") and not self.model.training:
        #    avg = self._eval_loss_ori_sum / max(1, self._eval_loss_ori_count)
        #    logs["eval_loss_ori"] = avg

            # reset
        #    self._eval_loss_ori_sum = 0.0
        #    self._eval_loss_ori_count = 0

        if hasattr(self, "_last_batch_beta"):
            logs["beta"] = float(self._last_batch_beta.item())

        if hasattr(self, "_last_margin"):
            logs["margin"] = float(self._last_margin.item())

        if hasattr(self, "_last_batch_mean_m"):
            logs["batch_mean_m"] = float(self._last_batch_mean_m.item())

        if hasattr(self, "_last_m0"):
            logs["current_m0"] = float(self._last_m0.item())

        super().log(logs)




def deterministic_train_split_only(records, exp_args: ExperimentArguments, output_dir: str):
    """
    Deterministic train subset sampling only, no validation split.
    """
    n = len(records)
    if n < 1:
        raise ValueError(f"Too few preference records: {n}.")

    rng = np.random.default_rng(exp_args.split_seed)
    perm = rng.permutation(n)

    # sample train subset from all data (or use all)
    train_size = int(n * exp_args.train_subset_ratio)
    train_size = max(exp_args.min_train_samples, train_size)
    train_size = min(train_size, n)

    train_idx = perm[:train_size]
    train_records = [records[i] for i in train_idx]

    os.makedirs(output_dir, exist_ok=True)

    split_info = {
        "total_records": n,
        "train_size": len(train_records),
        "val_size": 0,
        "train_subset_ratio": exp_args.train_subset_ratio,
        "split_seed": exp_args.split_seed,
        "min_train_samples": exp_args.min_train_samples,
        "note": "no validation split, all data used for training subset"
    }

    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    return Dataset.from_list(train_records)

def align_local():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, DataArguments, AttackArguments, ExperimentArguments)
    )
    model_args, training_args, data_args, attack_args, exp_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(exp_args.split_seed)
    np.random.seed(exp_args.split_seed)


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./github_repo/Meta_SecAlign/data",
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    os.makedirs(training_args.output_dir, exist_ok=True)
    print(f"\nOutput dir: {training_args.output_dir}\n")

    #preference_path = "./data/secalign++_training_data.json"
    # if not exp_args.preference_path:
    #     raise ValueError("You must provide --preference_path")
    #records_ori = json.load(open(exp_args.preference_path, "r", encoding="utf-8"))
    records_hard= json.load(open(exp_args.hard_path, "r", encoding="utf-8"))
    records=records_hard

    # data augmentation: randomly prepend chosen to rejected for half the data
    #records = augment_half_prepend_chosen(records)

    train_dataset = deterministic_train_split_only(
        records=records,
        exp_args=exp_args,
        output_dir=training_args.output_dir,
    )

    # ========= length statistics =========
    prompt_lengths = []
    prompt_and_chosen_lengths = []
    prompt_and_rejected_lengths = []

    for sample in records:
        prompt = sample["prompt"]
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")

        if not isinstance(prompt, str):
            continue

        # ===== prompt =====
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_lengths.append(len(prompt_ids))

        # ===== prompt + chosen =====
        full_chosen = prompt + chosen
        chosen_ids = tokenizer(full_chosen, add_special_tokens=False)["input_ids"]
        prompt_and_chosen_lengths.append(len(chosen_ids))

        # ===== prompt + rejected =====
        full_rejected = prompt + rejected
        rejected_ids = tokenizer(full_rejected, add_special_tokens=False)["input_ids"]
        prompt_and_rejected_lengths.append(len(rejected_ids))


    print('\n========== Length Statistics ==========')

    print('Prompt length (95%, 99%, 99.5%, 99.9%):',
        np.percentile(prompt_lengths, [95, 99, 99.5, 99.9]))

    print('Prompt+Chosen length (95%, 99%, 99.5%, 99.9%):',
        np.percentile(prompt_and_chosen_lengths, [95, 99, 99.5, 99.9]))

    print('Prompt+Rejected length (95%, 99%, 99.5%, 99.9%):',
        np.percentile(prompt_and_rejected_lengths, [95, 99, 99.5, 99.9]))

    print('=======================================\n')



    print(f"Train dataset size: {len(train_dataset)}")

    # force enable eval / save / early stop configs
    # ===== save =====
    training_args.save_strategy = "epoch"
    training_args.save_total_limit = 10

    # ===== no eval =====
    training_args.evaluation_strategy = "no"
    training_args.load_best_model_at_end = False
    training_args.do_eval = False
    training_args.max_grad_norm = 1.0


    # logging
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 1
    training_args.logging_first_step = True
    

    # disable external reporting like wandb
    if getattr(training_args, "report_to", None) is None:
        training_args.report_to = []
    else:
        training_args.report_to = []

    trainer_cls = {
        "dpo": BetaDPOTrainer,
        "kto": KTOTrainer,
        "orpo": ORPOTrainer,
    }[attack_args.alignment]

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    with open(os.path.join(training_args.output_dir, "finished.txt"), "w") as f:
        f.write("training finished\n")

    print("\nTraining finished. Best checkpoint has been loaded and final model is saved.\n")


if __name__ == "__main__":
    align_local()