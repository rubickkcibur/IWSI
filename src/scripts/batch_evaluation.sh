#!/bin/bash
for num in {2..10} 
do
    accelerate launch evaluation.py \
    --model_name_or_path "/mnt/lsk_cv/rubickjiang/proj_storage/adadf/huggingface/Meta-Llama-3-8B" \
    --peft_model_path "/mnt/lsk_cv/rubickjiang/proj_storage/adadf/peft_models/modelL-filter_strategy_Weighted-epoch_$((num-1))" \
    --data_path "/mnt/lsk_cv/rubickjiang/proj_storage/adadf/samples/llama3-8b_samples_gsm8k:7099_beam_1-temp_0.8-samples_1-15cands.jsonl" \
    --valid_data_path "gsm8k:7099" \
    --eval_data_path "gsm8k:test" \
    --data_filter_mode "Groundtruth" \
    --filter_base_model_path "/mnt/lsk_cv/rubickjiang/proj_storage/adadf/huggingface/Meta-Llama-3-8B" \
    --bf16 True \
    --output_dir "/mnt/lsk_cv/rubickjiang/proj_storage/adadf/peft_models" \
    --filter_model_lr 1e-5 \
    --uncertainty_th 1.0 \
    --num_train_epochs 1 \
    --filter_training_batch_size 8 \
    --valid_batch_size 16 \
    --filter_training_epochs 30 \
    --per_device_train_batch_size 6 \
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
    --model_max_length 800 \
    --lazy_preprocess False \
    --use_lora True \
    --gradient_checkpointing True
done