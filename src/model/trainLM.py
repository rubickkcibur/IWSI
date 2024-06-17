from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
import jsonlines
import copy
import random
import numpy as np
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from peft import PeftModel

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def qa_preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    processor,
    system_message: str = ""
) -> Dict:
    input_ids = []
    targets = []
    masks = []
    input_text = [processor(source) for source in sources]
    encoding = tokenizer(
        input_text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    input_ids = encoding["input_ids"]
    targets = copy.deepcopy(encoding["input_ids"])
    masks = encoding['attention_mask']
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=masks,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, data_processor, weight_extractor=None, groundtruth_extractor=None, question_extractor=None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        # sources = [example["conversations"] for example in raw_data]
        self.sources = raw_data
        if weight_extractor is not None:
            self.data_weights = [weight_extractor(d) for d in self.sources]
        else:
            self.data_weights = [0] * len(self.sources)
        if groundtruth_extractor is not None:
            self.groundtruths = [groundtruth_extractor(d) for d in self.sources]
        else:
            self.groundtruths = [0] * len(self.sources)
        if question_extractor is not None:
            self.questions = [question_extractor(d) for d in self.sources]
        else:
            self.questions = [0] * len(self.sources)
        data_dict = qa_preprocess(self.sources, tokenizer, max_len, data_processor)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids = self.input_ids[i],
            labels = self.labels[i],
            attention_mask = self.attention_mask[i],
            weight = self.data_weights[i],
            groundtruth = self.groundtruths[i],
            question = self.questions[i]
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len, weights
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    rank0_print("Loading data...")
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
    # train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(
        train_data, 
        tokenizer=tokenizer, 
        max_len=max_len, 
        data_processor=lambda x: "Q: " + x["question"] + tokenizer.eos_token + "A: " + x["answer"]
    )

    eval_dataset=None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def build_model(model_args, training_args, lora_args, logger):
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # gptq_config = GPTQConfig(
    #     bits=8, disable_exllama=True, tokenizer=tokenizer, dataset="c4"
    # )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        # quantization_config=gptq_config,
        torch_dtype=torch.bfloat16,
    )
    
    if len(model_args.peft_model_path) > 0:
        logger.info('loading peft model from {}'.format(model_args.peft_model_path))
        model = PeftModel.from_pretrained(model, model_args.peft_model_path)
        model.merge_and_unload()
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        return model, tokenizer
    if training_args.use_lora:
        # modules_to_save = None # q_lora
        modules_to_save = ["wte", "lm_head"] #lora
        logger.info("use peft!")
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            # target_modules=lora_args.lora_target_modules,
            # target_modules = ["q_proj","k_proj","v_proj","o_proj","down_proj","gate_proj","up_proj"],
            target_modules = ["q_proj","v_proj","down_proj","gate_proj","up_proj"],
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            # modules_to_save=None  # This argument serves for adding new tokens.
        )
        
        # model = prepare_model_for_kbit_training(
        #     model, use_gradient_checkpointing=training_args.gradient_checkpointing
        # ) #q_lora
        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            # model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    return model, tokenizer


def trainL(training_args, lora_args, model_args, model, tokenizer, train_dataset, eval_dataset=None):
    global local_rank
    logger = get_logger(__name__)
    num_training_steps = training_args.num_train_epochs * len(train_dataset)
    num_training_steps = int(num_training_steps)
    accelerate = Accelerator()
    device = accelerate.device
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # my train loop
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=training_args.per_device_train_batch_size, 
    )
    model, train_dataloader, optimizer, lr_scheduler = accelerate.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )

    pbar = tqdm.tqdm(total=len(train_dataloader)*int(training_args.num_train_epochs), disable=(not accelerate.is_local_main_process))
    print(training_args.num_train_epochs)
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        total_loss = []
        for batch in train_dataloader:
            outputs = model(
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
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, model_args.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view(batch_size, -1)
            loss = torch.mul(weights.unsqueeze(-1), loss)
            loss = torch.mean(loss)
            accelerate.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if accelerate.sync_gradients:
                pbar.update(batch_size)
            total_loss.append(loss.detach().item())
        accelerate.print("total loss: {}".format(sum(total_loss)/len(total_loss) if len(total_loss) > 0 else 0))
    accelerate.wait_for_everyone()
    accelerate.end_training()
