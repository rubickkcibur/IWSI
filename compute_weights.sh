#!/bin/bash
export NCCL_IB_DISABLE=1 
export NCCL_P2P_DISABLE=1

MODEL_PATH="" # base model
DATA_PATH="" # self-generated samples, jsonl file path 
DATASET_NAME="gsm8k"
OUTPUT_DIR="" # not used
TEMP_PATH="" # data output dir
VALID_DATA_PATH="" # valid data path
PEFT_MODEL_PATH="" # not used

# record weights for RM-filter
accelerate launch RM_weight.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --peft_model_path "" \
  --dataset_name "$DATASET_NAME" \
  --valid_data_path "" \
  --temp_data_path "$TEMP_PATH" \
  --eval_data_path "" \
  --data_filter_mode "None" \
  --filter_base_model_path "" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 0.8 \
  --num_train_epochs 5 \
  --filter_training_batch_size 8 \
  --valid_batch_size 16 \
  --filter_training_epochs 30 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 32 \
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
  --model_max_length 400 \
  --lazy_preprocess False \
  --use_lora True \
  --gradient_checkpointing True

# record weights for self-filter
accelerate launch self_weight.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --peft_model_path "" \
  --dataset_name "$DATASET_NAME" \
  --valid_data_path "" \
  --temp_data_path "$TEMP_PATH" \
  --eval_data_path "" \
  --data_filter_mode "None" \
  --filter_base_model_path "" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 0.8 \
  --num_train_epochs 4 \
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
  --model_max_length 400 \
  --lazy_preprocess False \
  --use_lora True \
  --gradient_checkpointing True

# record weight for IWSI
# record valid loss
accelerate launch record_loss.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "" \
  --peft_model_path "$PEFT_MODEL_PATH" \
  --dataset_name "$DATASET_NAME" \
  --valid_data_path "$VALID_DATA_PATH" \
  --temp_data_path "$TEMP_PATH" \
  --eval_data_path "" \
  --data_filter_mode "None" \
  --filter_base_model_path "" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 0.8 \
  --num_train_epochs 5 \
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
  --model_max_length 400 \
  --lazy_preprocess False \
  --use_lora True \
  --gradient_checkpointing True

# record loss for self-generated data
accelerate launch record_loss.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --peft_model_path "$PEFT_MODEL_PATH" \
  --dataset_name "$DATASET_NAME" \
  --valid_data_path "" \
  --temp_data_path "$TEMP_PATH" \
  --eval_data_path "" \
  --data_filter_mode "None" \
  --filter_base_model_path "" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 0.8 \
  --num_train_epochs 5 \
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
  --model_max_length 400 \
  --lazy_preprocess False \
  --use_lora True \
  --gradient_checkpointing True

# compute weight
python record_weights.py \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --peft_model_path "$PEFT_MODEL_PATH" \
  --dataset_name "$DATASET_NAME" \
  --valid_data_path "" \
  --temp_data_path "$TEMP_PATH" \
  --eval_data_path "" \
  --data_filter_mode "None" \
  --filter_base_model_path "" \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --filter_model_lr 1e-5 \
  --uncertainty_th 0.8 \
  --num_train_epochs 5 \
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
  --model_max_length 400 \
  --lazy_preprocess False \
  --use_lora True \
  --gradient_checkpointing True
exit 0