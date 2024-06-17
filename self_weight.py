import copy
from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np
from src.utils.utils import derive_num_from_answer, format_ground_truth_answer, derive_ratings_from_answer
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
from accelerate.utils import set_seed, gather_object
import logging
import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
import warnings
import json

warnings.filterwarnings("ignore")

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
    cand_num = 0
    with jsonlines.open(data_args.data_path, "r") as reader:
        for idx, obj in enumerate(reader):
            question = obj["question"]
            candidates = obj["candidates"]
            cands_weight = weights[idx]
            assert len(candidates) == len(cands_weight)
            cand_num = len(candidates)
            for i in range(len(candidates)):
                train_data.append({
                    "question": question,
                    "answer": candidates[i],
                    "weight": cands_weight[i]
                })
    # if data_args.dataset_name == "aqua_rat":
    #     train_data = train_data[:2000*15]
    train_dataset = SupervisedDataset(
        train_data, 
        tokenizer=tokenizer, 
        max_len=max_len, 
        data_processor=lambda x: "Q: " + x["question"] + "\n" + "A: " + tokenizer.eos_token + x["answer"],
        weight_extractor=lambda x: x["weight"]
    )
    return train_dataset, cand_num

def self_weight():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()
    device = accelerator.device
    logger.info('Loading causal model...')
    modelL = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch.bfloat16
        )
    if len(model_args.peft_model_path) > 0:
        logger.info("loading peft weights from{}".format(model_args.peft_model_path))
        modelL = PeftModel.from_pretrained(modelL, model_args.peft_model_path)
        modelL.merge_and_unload()
    tokenizerL = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        model_max_length=training_args.model_max_length,
        use_fast=False,
        padding_side = "left")
    tokenizerL.pad_token_id = tokenizerL.eos_token_id

    logger.info('Loading training&validation data...')
    dump_file_name = ""

    data_weights = get_data_weight(data_args, model=None)
    train_dataset, cand_num = load_train_data(tokenizerL, data_args, training_args.model_max_length, data_weights)
    train_dataset = train_dataset.sources
    prompts = [
        COT_EXAMPLES["self"] + "Q: {}\n".format(x["question"]) + "A: {}\n".format(x["answer"]) + "Score: " for x in train_dataset
    ]
    dump_file_name = "self_weight.json"

    tokenizerL.padding_side="left"

    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]
        tokenizer.padding_side="left"     
        for prompt_batch in batches:
            batches_tok.append(
                tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=True, 
                    max_length=training_args.model_max_length,
                    add_special_tokens=True).to(device) 
                )
        return batches_tok
    
    propmt_length = len(prompts)
    modelL.eval()
    modelL.to(device)
    accelerator.wait_for_everyone()

    logger.info('Start Loss computing...')
    with accelerator.split_between_processes(prompts) as prompts:
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizerL, batch_size=training_args.per_device_eval_batch_size)
        pbar = tqdm.tqdm(total=len(prompt_batches), disable=(not accelerator.is_local_main_process))

        for prompts_tokenized in prompt_batches:
            with torch.no_grad():
                outputs_tokenized=modelL.generate(**prompts_tokenized, max_length=training_args.model_max_length, num_return_sequences=1, temperature=0.7, pad_token_id=tokenizerL.eos_token_id)

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_tokenized ])
            outputs=tokenizerL.batch_decode(outputs_tokenized)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
            if accelerator.is_local_main_process:
                pbar.update(1)
            torch.cuda.empty_cache()
        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly
    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        total_results = []
        total_results = []
        for r in results_gathered:
            total_results += r["outputs"]
        total_results = [answer.split(tokenizerL.eos_token)[0] if tokenizerL.eos_token in answer else answer for answer in total_results]
        if len(total_results) > propmt_length:
            total_results = total_results[:propmt_length]
        total_results = [derive_ratings_from_answer(x)  for x in total_results]
        total_results = [min(max(0, float(x)), 10) if x is not None else 0 for x in total_results]
        logger.info("results length is {}".format(len(total_results)))
        total_results = [total_results[i:i+cand_num] for i in range(0, len(total_results), cand_num)]
        save_path = os.path.join(data_args.temp_data_path, data_args.dataset_name.replace("/", "_"))
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, dump_file_name), "w", encoding="utf8") as f:
            json.dump(total_results,f)

if __name__ == "__main__":
    seed = 114514
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    self_weight()
    

