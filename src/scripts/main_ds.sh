#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_HOME=$CONDA_PREFIX
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="Qwen/Qwen-7B" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="path_to_data"
DS_CONFIG_PATH="finetune/ds_config_zero2.json"

function usage() {
    echo '
Usage: bash finetune/finetune_lora_ds.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH]
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
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
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

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS main.py \
    --model_name_or_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
    --data_path "/aifs4su/users/rubickjiang/proj_storage/adadf/samples/gsm8k_self_generate_samples_beam_1-temp_0.8-samples_15-time_1712919217_train_7099.jsonl" \
    --bf16 True \
    --valid_data_path "gsm8k:7099" \
    --eval_data_path "gsm8k:test" \
    --data_filter_mode "Groundtruth" \
    --filter_base_model_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
    --output_dir "/aifs4su/users/rubickjiang/proj_storage/adadf/models" \
    --filter_model_lr 1e-5 \
    --uncertainty_th 1.0 \
    --filter_training_batch_size 8 \
    --valid_batch_size 16 \
    --filter_training_epochs 30 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed "/home/rubickjiang/adadf/dsconfig/ds_config_zero2.json"