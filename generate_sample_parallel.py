from transformers import Trainer, GPTQConfig, deepspeed, DataCollatorWithPadding, AdamW, get_scheduler
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
import torch
import random
import numpy as np
from src.utils.utils import derive_num_from_answer, derive_num_from_output, format_ground_truth_answer, get_extractors, get_qa_pair
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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
device = None

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="your_model_path")
    filter_base_model_path: str = field(default="")
    vocab_size: int = field(default=0)
    peft_model_path: str = field(default="")

@dataclass
class DataArguments:
    data_path: str = field(
        default="your_test_file/gsm8k_test.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    valid_data_path: str = field(
        default=None, metadata={"help": "valid data path, name:split"}
    )
    dataset_name: str = field(
        default=None
    )
    data_filter_mode: str = field(
        default="Consistency", metadata={"help": "Consistency, Groundtruth, Entropy, Weighted, Mixed, K-Mixed"}
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

def generate_sample_para():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    N_RETURN_SEQUENCES = 1
    beam_K = 1
    temperature = 1.1
    accelerator = Accelerator()
    os.makedirs(training_args.output_dir, exist_ok=True)
    parameter_appendix = "beam_{}-temp_{}-samples_{}-time_{}".format(beam_K, temperature, N_RETURN_SEQUENCES, os.environ["TIME_SUFFIX"])
    file_name = os.path.join(
        training_args.output_dir, 
        "llama3-8b_samples_{}_{}.jsonl".format(data_args.data_path.replace("/","_"),parameter_appendix))
    if accelerator.is_local_main_process:
        f = open(file_name,"w")
        f.close()

    device = accelerator.device

    logger.info('Loading peft model...')
    modelL = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    if len(model_args.peft_model_path) > 0:
        logger.info("loading peft weights from{}".format(model_args.peft_model_path))
        modelL = PeftModel.from_pretrained(modelL, model_args.peft_model_path)
        modelL.merge_and_unload()
    tokenizerL = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side = "left")
    tokenizerL.pad_token_id = tokenizerL.eos_token_id
    name, split = data_args.data_path.split(":")
    if name == "facebook/anli":
        if split == "r2":
            dataset = load_dataset(name, split="train_r2[:2000]")
        else:
            dataset = load_dataset(name, split="train_r1[:2000]")
    elif split.isdigit():
        dataset = load_dataset(name, data_dir="main", split="train[:{}]".format(split)) if name in ["gsm8k"] else load_dataset(name, split="train[:{}]".format(split))
    else:
        dataset = load_dataset(name, split=split)

    dp, ge, qe = get_extractors(data_args.dataset_name)

    train_dataset = SupervisedDataset(
        dataset, 
        tokenizer=tokenizerL, 
        max_len=training_args.model_max_length, 
        data_processor=dp,
        groundtruth_extractor=ge,
        question_extractor=qe
    )
    train_dataset = train_dataset.sources
    questions = [qe(x) for x in train_dataset]
    prompts = [dp(x) for x in train_dataset]
    answers = [ge(x) for x in train_dataset]
    

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
    modelL.eval()
    modelL.to(device)
    accelerator.wait_for_everyone()

    with accelerator.split_between_processes(prompts) as prompts:
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizerL, batch_size=training_args.per_device_eval_batch_size)
        pbar = tqdm.tqdm(total=len(prompt_batches), disable=(not accelerator.is_local_main_process))

        for prompts_tokenized in prompt_batches:
            outputs_tokenized=modelL.generate(
                **prompts_tokenized, 
                max_length=training_args.model_max_length, 
                num_return_sequences=N_RETURN_SEQUENCES, 
                num_beams=beam_K, 
                temperature=temperature,
                pad_token_id=tokenizerL.eos_token_id,
                do_sample=True)

            # remove prompt from gen. tokens
            outputs_list = []
            for batch_i in range(len(prompts_tokenized["input_ids"])):
                prefix_len = len(prompts_tokenized["input_ids"][batch_i])
                for j in range(N_RETURN_SEQUENCES):
                    outputs_list.append(outputs_tokenized[batch_i*N_RETURN_SEQUENCES + j][prefix_len:])


            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_list ])
            outputs=tokenizerL.batch_decode(outputs_list, skip_special_tokens=True)
            outputs = [outputs[i:i+N_RETURN_SEQUENCES] for i in range(0, len(outputs), N_RETURN_SEQUENCES)]

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
            if accelerator.is_local_main_process:
                pbar.update(1)

        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly
    results_gathered=gather_object(results)
    if accelerator.is_main_process:
        total_results = []
        for r in results_gathered:
            total_results += r["outputs"]
        # total_results = [answer.split("</s>")[0] if "</s>" in answer else answer for answer in total_results]
        assert len(total_results) == len(questions)
        generated_samples = []
        for i in range(len(questions)):
            cands = [get_qa_pair(questions[i], cand)[1] for cand in total_results[i]]
            generated_samples.append({
                "question": questions[i],
                "candidates": cands,
                "ground_truth": answers[i]
            })
        with jsonlines.open(file_name,"a") as writer:
            for sample in generated_samples:
                writer.write(sample)

if __name__ == "__main__":
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    generate_sample_para()
    

