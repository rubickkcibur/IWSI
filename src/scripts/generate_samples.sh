#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p debug
#SBATCH --qos=low
#SBATCH -J generate_sample
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

python -W ignore -u sample_generate.py \
    --dataset_name "gsm8k" \
    --dataset_path "gsm8k:main" \
    --storage_path "/aifs4su/users/rubickjiang/proj_storage/adadf/samples" \
    --model_path "/aifs4su/users/rubickjiang/.cache/huggingface/Llama-2-7b-hf" \
    --return_samples 15 \
    --beam_K 1 \
    --temperature 0.8 \
    --max_length 800 \
    --batch_size 4 \
    --gpu_id 0 \
    --val_portion 0.05 
