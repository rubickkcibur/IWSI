#!/bin/bash

#varibles
#peft_model_path
#mlp_store_path

# export CUDA_VISIBLE_DEVICES=0
PEFT_MODEL_PATH=""
MLP_STORE_PATH=""
# for raw llama: /aifs4su/users/rubickjiang/proj_storage/adadf/samples/gsm8k_self_generate_samples_beam_1-temp_0.8-samples_15-time_1712919217_train_7099.jsonl
# for finetuned llama: /aifs4su/users/rubickjiang/proj_storage/adadf/samples/self_generate_samples_train_on_valid_gsm8k:7099_beam_1-temp_0.8-samples_15-time_1713504414.jsonl
DATA_PATH=""
for num in {1..5}  
do   
    export ADAEPOCH=$num
    if [ $num -eq 1 ]; then
        PEFT_MODEL_PATH=""
        MLP_STORE_PATH=""
    else
        PEFT_MODEL_PATH="/aifs4su/users/rubickjiang/proj_storage/adadf/models/modelL-filter_strategy_Weighted-epoch_$((num-1))"
        MLP_STORE_PATH="/aifs4su/users/rubickjiang/proj_storage/adadf/models/modelF-filter-time_$((num-1))"
    fi
    # compute train loss
    accelerate launch record_loss.py \
    --model_name_or_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
    --data_path "$DATA_PATH" \
    --peft_model_path "$PEFT_MODEL_PATH" \
    --valid_data_path "" \
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
    --model_max_length 500 \
    --lazy_preprocess False \
    --use_lora \
    --gradient_checkpointing 
    # compute train loss
    accelerate launch record_loss.py \
    --model_name_or_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
    --data_path "" \
    --peft_model_path "$PEFT_MODEL_PATH" \
    --valid_data_path "gsm8k:7099" \
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
    --per_device_eval_batch_size 16 \
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
    --model_max_length 500 \
    --lazy_preprocess False \
    --use_lora \
    --gradient_checkpointing 

    # train filter
    accelerate launch train_filter.py \
      --model_name_or_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
      --data_path "$DATA_PATH" \
      --temp_data_path "/aifs4su/users/rubickjiang/proj_storage/adadf/temp" \
      --mlp_store_path "$MLP_STORE_PATH" \
      --valid_data_path "gsm8k:7099" \
      --eval_data_path "gsm8k:test" \
      --data_filter_mode "None" \
      --filter_base_model_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
      --bf16 True \
      --output_dir "/aifs4su/users/rubickjiang/proj_storage/adadf/models" \
      --filter_model_lr 1e-5 \
      --uncertainty_th 1.0 \
      --num_train_epochs 5 \
      --filter_training_batch_size 32 \
      --valid_batch_size 16 \
      --filter_training_epochs 8 \
      --per_device_train_batch_size 6 \
      --per_device_eval_batch_size 1 \
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
      --model_max_length 500 \
      --lazy_preprocess False \
      --use_lora \
      --gradient_checkpointing 

    # compute weight
    MLP_STORE_PATH="/aifs4su/users/rubickjiang/proj_storage/adadf/models/modelF-filter-time_${num}"
    accelerate launch record_weights.py \
      --model_name_or_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
      --data_path "$DATA_PATH" \
      --peft_model_path "$PEFT_MODEL_PATH" \
      --valid_data_path "" \
      --mlp_store_path "$MLP_STORE_PATH" \
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
      --model_max_length 500 \
      --lazy_preprocess False \
      --use_lora \
      --gradient_checkpointing 

    # train modelL
    accelerate launch baseline.py \
    --model_name_or_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
    --data_path "$DATA_PATH" \
    --peft_model_path "$PEFT_MODEL_PATH" \
    --valid_data_path "gsm8k:7099" \
    --eval_data_path "gsm8k:test" \
    --data_filter_mode "Weighted" \
    --filter_base_model_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
    --bf16 True \
    --output_dir "/aifs4su/users/rubickjiang/proj_storage/adadf/models" \
    --temp_data_path "/aifs4su/users/rubickjiang/proj_storage/adadf/temp" \
    --filter_model_lr 1e-5 \
    --uncertainty_th 1.0 \
    --num_train_epochs 5 \
    --filter_training_batch_size 8 \
    --valid_batch_size 16 \
    --filter_training_epochs 30 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
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
    --model_max_length 500 \
    --lazy_preprocess False \
    --use_lora \
    --gradient_checkpointing 
done
# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json