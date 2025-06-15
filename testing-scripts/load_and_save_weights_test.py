from importlib.metadata import version
import torch
from gpt import *
import tiktoken
from utils import generate_text_simple, text_to_token_ids, token_ids_to_text, create_dataloader_v1, calc_loss_loader, train_model_simple, set_plt_params, plot_eval
import matplotlib.pyplot as plt
import numpy 
import tensorflow
import os 
import urllib.request
tokenizer = tiktoken.get_encoding("gpt2")
"""
Training our GPT model with the-verdixt.txt

"""

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

train_ratio = 0.9
split_idx = int(train_ratio*len(text_data))
train_data = text_data[: split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = True, 
    shuffle = True,
    num_workers = 0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = True, 
    shuffle = True,
    num_workers = 0
)


cpu = "cpu"
device = torch.device(cpu)

train_losses = []
test_losses = []

model = GPTModel(GPT_CONFIG_124M)
model = model.to(device)
with torch.no_grad(): 
    #since we're not training, gradients are disabled
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("training loss", train_loss)
print("test loss/val loss", val_loss)
# it works! 

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
num_epochs = 5
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, 
    num_epochs = num_epochs, eval_freq = 5, eval_iter = 5,
    start_context = "Every effort moves you ", tokenizer = tokenizer)

torch.save(model.state_dict(), "test_model.pth")

model2 = GPTModel(GPT_CONFIG_124M)
model2.load_state_dict(torch.load("test_model.pth"), map_location = device)
model.eval()

