# LocalAlign

LocalAlign is a lightweight research repository for training and evaluating local alignment methods against prompt injection and instruction hijacking in large language models.

The codebase focuses on three parts:

- preference data generation for alignment training
- model training and variant experiments
- IID / OOD / SEP-style robustness evaluation

## Repository Structure

```text
.
├── config.py
├── generate_data.py
├── prepare_tokenizer.py
├── train.py
├── run_secalign_belta_Z_smooth.py
├── eva-iid.py
├── eva-ood.py
├── eva-sep-api.py
├── utils.py
└── data/
```

## Main Scripts

- `generate_data.py`: generate preference-style training data
- `prepare_tokenizer.py`: prepare tokenizer/chat template assets
- `train.py`: supervised or preference-based training entry
- `run_secalign_belta_Z_smooth.py`: run beta-DPO style SecAlign experiments
- `eva-iid.py`: in-distribution evaluation
- `eva-ood.py`: out-of-distribution robustness evaluation
- `eva-sep-api.py`: API-based SEP / attack success evaluation

## Environment

Recommended:

- Python 3.10+
- PyTorch
- `transformers`
- `trl`
- `datasets`
- `peft`
- `numpy`
- `tqdm`
- `openai`
- `vllm` or a compatible local inference backend

Example installation:

```bash
pip install torch transformers trl datasets peft numpy tqdm openai vllm
```

## Quick Start

Generate training data:

```bash
python generate_data.py
```

Train a model:

```bash
python train.py \
  --model_name_or_path <base-model> \
  --data_path <training-data.json> \
  --output_dir outputs/train_run
```

Run OOD evaluation:

```bash
python eva-ood.py \
  --base-path <base-model> \
  --output-dir outputs/ood_eval
```

Run SEP evaluation on generated results:

```bash
python eva-sep-api.py \
  --generation_dir <generation-results-dir>
```

## Notes

- Some evaluation scripts expect benchmark files under `data/` or mirrored third-party resources referenced by default paths in the code.
- Several scripts assume access to GPU inference and model checkpoints.
- Before running large experiments, it is a good idea to inspect the default arguments in each script and adjust paths, model names, and attack settings for your setup.

## Status

This repository is organized as an experiment-oriented codebase rather than a polished package. It is best suited for reproducing or extending local alignment and prompt-injection robustness experiments.
