from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from src.utils.constants import COT_EXAMPLES
from tqdm import tqdm
from src.utils.utils import derive_num_from_answer, derive_num_from_output
import random
import copy

def test_batch_loss(model, tokenizer, questions, answers, max_length):
    sentence = ["Q: " + q + tokenizer.eos_token + "A: " + a for q, a in zip(questions, answers)]
    with torch.no_grad():
        encoding = tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
    labels = copy.deepcopy(encoding["input_ids"])
    output = model(
        **encoding,
        labels=labels
    )
    loss = output.loss
    return float(loss.cpu().detach().numpy())

def test_loss(model, tokenizer, question, answer, max_length):
    sentence = "Q: " + question + tokenizer.eos_token + "A: " + answer
    with torch.no_grad():
        encoding = tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
    labels = copy.deepcopy(encoding["input_ids"])
    output = model(
        **encoding,
        labels=labels
    )
    loss = output.loss
    return float(loss.cpu().detach().numpy())

def evalauation(model, tokenizer, data_args, training_args, test_dataset):
    batch_size = training_args.valid_batch_size

    def get_integer(n):
        f = float(n)
        if f > 1e10:
            f = 1e10
        elif f < 1e-10:
            f = 1e-10
        return int(f)
    
    # def my_collate(d):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        )
    acc = 0
    total_answers = []
    total_outputs = []
    with torch.no_grad():
        with tqdm(total=len(test_dataset), desc="evaluation on testset") as bar: 
            for item in test_dataloader:
                input_ids = item["input_ids"]
                questions = [str(d) for d in item["question"]]
                answers = item["groundtruth"]
                input_ids = input_ids.to(device)
                generated_outputs = model.generate(input_ids=input_ids, max_length = 800, num_return_sequences=1, temperature=0.3)
                generated_texts = [tokenizer.decode(generated_output, skip_special_tokens=True) for generated_output in generated_outputs]
                outputs = generated_texts
                
                outputs = [derive_num_from_output(t.split(q)[-1]) for q,t in zip(questions, outputs)]
                total_outputs += outputs

                acc += sum([1 if o is not None and get_integer(a) == get_integer(o) else 0 for a, o in zip(answers, outputs)])

    print(acc / len(test_dataset))
    return acc / len(test_dataset)
