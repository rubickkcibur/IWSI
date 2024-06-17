#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_PATH=""
PEFT_MODEL=""
DATASET_NAME="gsm8k"
SPLIT=7099
OUTPUT_DIR=""
# export CUDA_VISIBLE_DEVICES=0
export TIME_SUFFIX=$(date +'%m-%d-%H-%M-%S')
accelerate launch generate_sample_parallel.py \
  --model_name_or_path "$MODEL_PATH" \
  --peft_model_path "$PEFT_MODEL" \
  --data_path "$DATASET_NAME:$SPLIT" \
  --dataset_name "$DATASET_NAME" \
  --valid_data_path "" \
  --eval_data_path "" \
  --data_filter_mode "Groundtruth" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 1.0 \
  --num_train_epochs 1 \
  --filter_training_batch_size 8 \
  --valid_batch_size 16 \
  --filter_training_epochs 30 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
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
  --model_max_length 1024 \
  --lazy_preprocess False \
  --use_lora True \
  --gradient_checkpointing True
exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json