import os
import torch
from torch import nn
import transformers
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import random
from tqdm import tqdm

m = transformers.LlamaForCausalLM
class FilterModel(nn.Module):
    def __init__(self, model_args, training_args, logger):
        self.logger = logger
        llama_path = model_args.filter_base_model_path
        max_length = training_args.model_max_length
        lr = training_args.filter_model_lr
        batch_size = training_args.filter_training_batch_size
        self.mlp_path = model_args.mlp_store_path
        super(FilterModel, self).__init__()
        self.max_length = max_length
        self.lr = lr
        self.llama_config = AutoConfig.from_pretrained(llama_path)
        self.llama = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.bfloat16)
        self.llama.eval()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained(llama_path, padding_side = "left")
        self.mlp = nn.Sequential(
            nn.Linear(self.llama_config.hidden_size, self.llama_config.hidden_size),
            nn.Linear(self.llama_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.ReLU() #做截断吗
            # nn.Sigmoid()
        )
        self.batch_size = batch_size
        self.initialize()

    def initialize(self):
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.llama.to(self.device)
        # self.mlp.to(self.device)
        if len(self.mlp_path) > 0:
            self.logger.info("load checkpoint from {}".format(self.mlp_path))
            self.mlp.load_state_dict(self.load_mlp(self.mlp_path))
        else:
            for name, param in self.mlp.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
        # self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr) 

    def get_sentence_vector(self, inputs):
        with torch.no_grad():
            # inputs = self.tokenizer(input, return_tensors='pt', padding="max_length", truncation=True, max_length=self.max_length)
            # inputs = {k: v.to(self.device) for k,v in inputs.items()}
            outputs = self.llama(
                **inputs,
                output_hidden_states = True
            )
            hidden_states = outputs.hidden_states
        hidden_states = hidden_states[-1]
        return torch.mean(hidden_states, dim=-2).detach()

    def predict(self, batch_sentence):
        hidden_states = self.get_sentence_vector(batch_sentence)
        hidden_states = hidden_states.to(torch.float)
        score = self.mlp(hidden_states)
        score = torch.squeeze(score)
        return score.cpu().detach().numpy()
    
    def forward(self, batch_sentence):
        hidden_states = self.get_sentence_vector(batch_sentence)
        hidden_states = hidden_states.to(torch.float)
        score = self.mlp(hidden_states)
        return score

    def DIWLoss(self, val_avg_loss, batch_sentence, train_avg_loss_vector):
        weight = self.forward(batch_sentence)
        train_avg_loss = torch.mul(weight, train_avg_loss_vector)
        train_avg_loss = torch.mean(train_avg_loss)
        Dloss = (val_avg_loss - train_avg_loss)**2
        return Dloss
    
    def save_mlp(self, path):
        torch.save(self.mlp.state_dict(), os.path.join(path, "mlp.pth"))
    
    def load_mlp(self, path):
        return torch.load(os.path.join(path, "mlp.pth"))
    
    # def train_epochs(self, dataset, data_losses, avg_val_loss, epochs):
    #     assert len(dataset) == len(data_losses)
    #     length = len(dataset)
    #     self.mlp.train()
    #     for e in tqdm(range(epochs), desc="train filter model"):
    #         id_list = list(range(len(dataset)))
    #         random.shuffle(id_list)
    #         losses = []
    #         for i in range(0, length, self.batch_size):
    #             batch_id = id_list[i:i+self.batch_size] if i+self.batch_size < length else id_list[i:]
    #             batch_sentence = [
    #                 "Q: " + dataset[k]["question"] + self.tokenizer.eos_token + "A: " + dataset[k]["answer"] for k in batch_id
    #             ]
    #             # batch_loss = torch.tensor([data_losses[k] for k in batch_id]).to(self.device)
    #             self.optimizer.zero_grad()
    #             loss = self.DIWLoss(avg_val_loss, batch_sentence, batch_loss)
    #             loss.backward()
    #             losses.append(float(loss.cpu().detach().numpy()))
    #             self.optimizer.step()
    #         print("loss in epoch {} is {}".format(e, sum(losses)/len(losses)))




    
    


