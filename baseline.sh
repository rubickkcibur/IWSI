#!/bin/bash
MODEL_PATH="" # base model
DATA_PATH="" # self-generated samples, the jsonl file
DATASET_NAME="gsm8k"
VALID_DATA_PATH="" # not used
OUTPUT_DIR="" # model checkpoint save dir
TEMP_PATH="" # data dir. This should contains the "weight.json" if u set mode to "Mixed" or "K-Mixed"

accelerate launch baseline.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --peft_model_path "" \
  --dataset_name "$DATASET_NAME" \
  --temp_data_path "$TEMP_PATH" \
  --valid_data_path "$VALID_DATA_PATH" \
  --eval_data_path "" \
  --data_filter_mode "K-Mixed" \
  --filter_base_model_path "" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 0.79 \
  --num_train_epochs 4 \
  --filter_training_batch_size 8 \
  --valid_batch_size 16 \
  --filter_training_epochs 30 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
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
  --model_max_length 400 \
  --lazy_preprocess False \
  --use_lora True \
  --gradient_checkpointing True

exit 0