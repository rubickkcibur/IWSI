from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np
from src.utils.utils import derive_num_from_answer
from src.utils.constants import COT_EXAMPLES
from src.model.trainLM import SupervisedDataset, trainL, build_model, make_supervised_data_module
from src.data.filter_data import get_data_weight
from src.model.filterLM import FilterModel
from src.utils.evaluation import test_loss, test_batch_loss, evalauation
from datasets import load_dataset
import jsonlines
from tqdm import tqdm
import time
import os
import sys
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
import logging
import tqdm
from torch.utils.data import Dataset, DataLoader
import json
'''
this script is to train LLM on filtered generated data. The two major usage is:
1. train baseline methods
2. train modelL in ADADF
'''

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
device = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    filter_base_model_path: str = field(default="")
    vocab_size: int = field(default=0)
    peft_model_path: str = field(default="")

@dataclass
class DataArguments:
    data_path: str = field(
        default="", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    valid_data_path: str = field(
        default=None, metadata={"help": "valid data path, name:split"}
    )
    temp_data_path: str = field(
        default=None
    )
    dataset_name: str = field(
        default=None
    )
    data_filter_mode: str = field(
        default="Consistency", metadata={"help": "Consistency, Groundtruth, Entropy, Weighted"}
    )
    lazy_preprocess: bool = False
    uncertainty_th: float = field(
        default=1.0
    )
    special_weight_path: str = field(
        default="", metadata={"help": "special weight"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=800,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    filter_training_batch_size: int = field(default=8)
    valid_batch_size: int = field(default=16)
    filter_training_epochs: int = field(default=10)
    filter_model_lr: float = field(
        default=1e-3
    )

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

def load_train_data(tokenizer: transformers.PreTrainedTokenizer, data_args, max_len, weights):
    train_data = []
    with jsonlines.open(data_args.data_path, "r") as reader:
        for idx, obj in enumerate(reader):
            question = obj["question"]
            candidates = obj["candidates"]
            cands_weight = weights[idx]
            assert len(candidates) == len(cands_weight)
            for i in range(len(candidates)):
                train_data.append({
                    "question": question,
                    "answer": candidates[i],
                    "weight": cands_weight[i]
                })
    train_dataset = SupervisedDataset(
        train_data, 
        tokenizer=tokenizer, 
        max_len=max_len, 
        data_processor=lambda x: "Q: " + x["question"] + "\n" + "A: " + tokenizer.eos_token + x["answer"],
        weight_extractor=lambda x: x["weight"]
    )
    return train_dataset


def baseline_main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    output_dir = "modelL-filter_strategy_{}-ua_{}-time_{}".format(data_args.data_filter_mode, data_args.uncertainty_th, int(time.time()))
    if data_args.dataset_name is None or data_args.dataset_name == "":
        detailed_output_dir = os.path.join(training_args.output_dir, output_dir)
    else:
        detailed_output_dir = os.path.join(training_args.output_dir, data_args.dataset_name.replace("/", "_"), output_dir)
    config = ProjectConfiguration(project_dir=detailed_output_dir, logging_dir="testfolder")
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps, log_with="tensorboard", project_config=config)
    if accelerator.is_local_main_process:
        os.makedirs(detailed_output_dir, exist_ok=True)

    accelerator.init_trackers("mode_{}".format(data_args.data_filter_mode))
    device = accelerator.device

    logger.info('Initializing model...')

    modelL, tokenizerL = build_model(model_args, training_args, lora_args, logger)
    model_args.vocab_size = modelL.config.vocab_size
    modelL.to(device)

    logger.info('Loading training&evaluation data...')
    data_weights = get_data_weight(data_args, model=None)
    if accelerator.is_local_main_process:
        if data_args.data_filter_mode in ["K-Mixed", "Entropy", "RM", "Self"]:
            path = os.path.join(data_args.temp_data_path, data_args.dataset_name.replace("/","_"), "{}_{}_weights.json".format(data_args.data_filter_mode, data_args.uncertainty_th))
        else:
            path = os.path.join(data_args.temp_data_path, data_args.dataset_name.replace("/","_"), "{}_weights.json".format(data_args.data_filter_mode))
        with open(path, "w") as f:
            json.dump(data_weights, f)
    train_dataset = load_train_data(tokenizerL, data_args, training_args.model_max_length, data_weights)
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=training_args.per_device_train_batch_size, 
    )
    epochs = training_args.num_train_epochs
    train_steps = epochs * len(train_dataloader)
    optimizer = AdamW(modelL.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=train_steps,
    )
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    logger.info('accelerator preparing...')

    modelL, train_dataloader,  optimizer, lr_scheduler = accelerator.prepare(modelL, train_dataloader, optimizer, lr_scheduler)

    iter = 0
    for epoch in range(int(epochs)):
        logger.info('=' * 10 + 'Start training' + '=' * 10)
        modelL.train()
        total_loss = 0
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=(not accelerator.is_local_main_process))
        with accelerator.accumulate(modelL):
            for i, batch in pbar:
                outputs = modelL(
                    input_ids = batch["input_ids"],
                    labels = batch["labels"],
                    attention_mask = batch["attention_mask"],
                )
                logits = outputs.get("logits")
                labels = batch["labels"]
                weights = batch["weight"]
                # Shift so that tokens < n predict n
                batch_size = logits.shape[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, model_args.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                # shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss.view(batch_size, -1)
                loss = torch.mul(weights.unsqueeze(-1), loss)
                loss = torch.mean(loss)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.log({"training_loss": loss}, step=iter)
                iter += 1
        
                pbar.set_description(f"epoch {epoch + 1} iter {i}: train loss {loss.item():.5f}. lr {lr_scheduler.get_last_lr()[0]:e}")
                if accelerator.is_local_main_process:
                    total_loss += loss.item() 
        torch.cuda.empty_cache()
        logger.info(f'Total local training loss in epoch {epoch + 1} is: {total_loss}', main_process_only=True)
        logger.info('Saving checkpoint...')
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            unwraped_model = accelerator.unwrap_model(modelL)
            unwraped_model.save_pretrained(detailed_output_dir + "epoch_{}".format(epoch))

if __name__ == "__main__":
    seed = 114514
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    baseline_main()
    

