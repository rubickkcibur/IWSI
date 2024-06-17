#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_PATH="" # base model path
DATASET_NAME="gsm8k"
OUTPUT_DIR="" # not usedd
PEFT_MODEL="" # peft model path

# what matters: model_name_or_path, peft_model_path, eval_data_path, per_device_eval_batch_size
export SEED=114514
accelerate launch evaluation.py \
  --model_name_or_path "$MODEL_PATH" \
  --peft_model_path "$PEFT_MODEL" \
  --dataset_name "$DATASET_NAME" \
  --data_path "" \
  --valid_data_path "" \
  --eval_data_path "$DATASET_NAME:test" \
  --data_filter_mode "" \
  --filter_base_model_path "" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 1.0 \
  --num_train_epochs 1 \
  --filter_training_batch_size 8 \
  --valid_batch_size 16 \
  --filter_training_epochs 30 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 4 \
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
  --use_lora True \
  --gradient_checkpointing True

exit 0
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json