import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import generate_preference_dataset_directional, generate_preference_dataset,generate_preference_dataset_secalign,generate_preference_dataset_directional_ablation

# records = generate_preference_dataset_secalign(
#         preference_data_path='./data/secalign_training_data_final.json',
#         instruct_dataset="alpaca",
#         self_generated_response=False,
#         randomized_injection_position=False,
#         model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct'
#     )




def main():




    try:
        records = generate_preference_dataset_directional_ablation(
                preference_data_path='./data/ablation_prompt/secalign++_training_data_ratio_mix_1_hard_zerolap_wo_3.json',
                instruction_pair_path='./data/ablation_prompt/instruction_pair_mix_1_hard_zerolap_wo_3.json',
                instruct_dataset='alpaca',
                self_generated_response=True,
                randomized_injection_position=True,
                model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
                chosen_ratio=1,
                wo=3
            )
        print("Directional dataset generated successfully.")

    except Exception as e:
        print("Error in generate_preference_dataset_directional_qwen:")
        print(e)



    try:
        records = generate_preference_dataset_directional_ablation(
                preference_data_path='./data/ablation_prompt/secalign++_training_data_ratio_mix_1_hard_zerolap_wo_6.json',
                instruction_pair_path='./data/ablation_prompt/instruction_pair_mix_1_hard_zerolap_wo_6.json',
                instruct_dataset='alpaca',
                self_generated_response=True,
                randomized_injection_position=True,
                model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
                chosen_ratio=1,
                wo=6
            )
        print("Directional dataset generated successfully.")

    except Exception as e:
        print("Error in generate_preference_dataset_directional_qwen:")
        print(e)


# records = generate_preference_dataset_directional(
#         preference_data_path='./data/secalign++_training_data_ratio_mix_0.5_zerolap.json',
#         instruction_pair_path='./data/instruction_pair_mix_0.5_zerolap.json',
#         instruct_dataset='alpaca',
#         self_generated_response=True,
#         randomized_injection_position=True,
#         model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
#         chosen_ratio=0.5,
#     )


# records = generate_preference_dataset_directional(
#         preference_data_path='./data/secalign++_training_data_ratio_mix_0_easy_zerolap.json',
#         instruction_pair_path='./data/instruction_pair_mix_0_easy_zerolap.json',
#         instruct_dataset='alpaca',
#         self_generated_response=True,
#         randomized_injection_position=True,
#         model_name_or_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
#         chosen_ratio=0,
#     )

if __name__ == "__main__":
    
    main()