from importlib.metadata import version
import torch
from gpt import *
import tiktoken
from utils import generate_text_simple, text_to_token_ids, token_ids_to_text, create_dataloader_v1
import matplotlib
import numpy
import tensorflow
import os 
import urllib.request
tokenizer = tiktoken.get_encoding("gpt2")

pkgs = ["matplotlib", 
        "numpy", 
        "tiktoken", 
        "torch",
        "tensorflow" # For OpenAI's pretrained weights
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

GPT_CONFIG_124M["context_length"] = 256 # lessen computational load

torch.manual_seed(123)
# ps - this includes bias = False and drop_rate = 0.1, nowadays droprate = 0 during LLM training
model = GPTModel(GPT_CONFIG_124M)
model.eval() # disable dropout during inference

start_context = "Every effort moves you"
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(start_context, tokenizer),
    max_new_tokens = 10, 
    context_size = GPT_CONFIG_124M["context_length"]
)

print("out:", token_ids_to_text(token_ids, tokenizer))



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

# print(text_data[:99]) testing if it worked
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

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

# sanity checks: 

if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for training loader, try to lower context length "
    "or increase training ratio. ")

if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the testing set. lower context length"
    "or decrease train_ratio")

# other test: 

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# testing mps for m1 mac
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")
# doesn't work...

just_mps = "mps"
mps_altered = "mps:0"

# just mps
device = torch.device(just_mps)
print(f"Using {device} device, but is it available? {torch.backends.mps.is_available()}")

mps_test = [torch.backends.mps.is_built(), torch.backends.mps.is_available()]
mps_test_name = ["built", "available"]

for n, t in zip(mps_test_name, mps_test):
    print("mps is", n, ":", t)

# mps altered
device = torch.device(mps_altered)
print(f"Using {device} device, but is it available? {torch.backends.mps.is_available()}")

mps_test = [torch.backends.mps.is_built(), torch.backends.mps.is_available()]
mps_test_name = ["built", "available"]

for n, t in zip(mps_test_name, mps_test):
    print("mps is", n, ":", t)
