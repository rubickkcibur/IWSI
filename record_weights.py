import copy
from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np
from src.utils.utils import derive_num_from_answer, format_ground_truth_answer
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
from accelerate import DistributedDataParallelKwargs
import random

'''
This script is to compute weights for generated data using inference of modelF
'''

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
    mlp_store_path: str = field(default="") #add

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
    train_dataset = SupervisedDataset(
        train_data, 
        tokenizer=tokenizer, 
        max_len=max_len, 
        data_processor=lambda x: "Q: " + x["question"] + "\n" + "A: " + tokenizer.eos_token + x["answer"],
        weight_extractor=lambda x: x["weight"]
    )
    return train_dataset, cand_num


def record_weight():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # no filter version
    with open(os.path.join(data_args.temp_data_path, data_args.dataset_name.replace("/","_"), "valid_loss.json"), "r", encoding="utf8") as f:
        valid_losses = json.load(f)
    with open(os.path.join(data_args.temp_data_path, data_args.dataset_name.replace("/","_"), "train_loss.json"), "r", encoding="utf8") as f:
        train_losses = json.load(f)

    avg_valid_loss = sum(valid_losses) / len(valid_losses)
    weights = []
    for cand_losses in train_losses:
        cand_weight = []
        for loss in cand_losses:
            if loss == 0:
                w = 0
            else:
                w = min(2, max(0, avg_valid_loss/loss if (loss > 0 and avg_valid_loss > 0) else 0))
            cand_weight.append(w)
        weights.append(cand_weight)
    with open(os.path.join(data_args.temp_data_path, data_args.dataset_name.replace("/","_"), "weight.json"), "w", encoding="utf8") as f:
        json.dump(weights,f)

    return

if __name__ == "__main__":
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    record_weight()
    

