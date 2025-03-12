import torch
from utils import train_model_simple, create_dataloader_v1
from gpt import *
import tiktoken
import os 
import urllib.request
tokenizer = tiktoken.get_encoding("gpt2")

mps = "mps"
device = torch.device(mps)
print(f"Using {device} device, but is it available?  - {torch.backends.mps.is_available()}")

torch.manual_seed(123)
# ps - this includes bias = False and drop_rate = 0.1, nowadays droprate = 0 during LLM training
GPT_CONFIG_124M["context_length"] = 256
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
model.eval() # disable dropout during inference

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

torch.manual_seed(11)
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


optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, 
    num_epochs = num_epochs, eval_freq = 5, eval_iter = 5,
    start_context = "Every effort moves you", tokenizer = tokenizer)

