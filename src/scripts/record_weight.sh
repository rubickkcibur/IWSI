#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="/home/incoming/LLM/llama2/llama2-7b" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/home/LAB/jiangcy/AdaDF/filtered_samples/gsm8k_train_llama2-7b_filtered.jsonl"

function usage() {
    echo '
Usage: bash finetune/finetune_lora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

# export CUDA_VISIBLE_DEVICES=0

accelerate launch record_weights.py \
  --model_name_or_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
  --data_path "/aifs4su/users/rubickjiang/proj_storage/adadf/samples/gsm8k_self_generate_samples_beam_1-temp_0.8-samples_15-time_1712919217_train_7099.jsonl" \
  --peft_model_path "/aifs4su/users/rubickjiang/proj_storage/adadf/models/modelL-filter_strategy_None-time_1713274337" \
  --valid_data_path "" \
  --mlp_store_path "/aifs4su/users/rubickjiang/proj_storage/adadf/models/modelF-filter-time_None" \
  --temp_data_path "/aifs4su/users/rubickjiang/proj_storage/adadf/temp" \
  --eval_data_path "gsm8k:test" \
  --data_filter_mode "None" \
  --filter_base_model_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
  --bf16 True \
  --output_dir "/aifs4su/users/rubickjiang/proj_storage/adadf/models" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 1.0 \
  --num_train_epochs 5 \
  --filter_training_batch_size 8 \
  --valid_batch_size 16 \
  --filter_training_epochs 30 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 24 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 800 \
  --lazy_preprocess False \
  --use_lora \
  --gradient_checkpointing 

# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json