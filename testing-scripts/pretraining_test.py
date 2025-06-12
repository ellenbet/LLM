from importlib.metadata import version
import torch
from gpt import *
import tiktoken
from utils import generate_text_simple, text_to_token_ids, token_ids_to_text, create_dataloader_v1, calc_loss_loader, train_model_simple, set_plt_params
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy 
import tensorflow
import os 
import urllib.request
tokenizer = tiktoken.get_encoding("gpt2")

pkgs = ["matplotlib", 
        "seaborn",
        "numpy", 
        "tiktoken", # For tokenizer library
        "torch",
        "tensorflow" # For OpenAI's pretrained weights
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

GPT_CONFIG_124M["context_length"] = 256 # lessen computational load
set_plt_params()
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
#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
#print(f"Using {device} device.")
# doesn't work...

#just_mps = "cpu"
cpu = "cpu"

# just mps
#device = torch.device(just_mps)
device = torch.device(cpu)
#print(f"Using {device} device, but is it available?  - {torch.backends.mps.is_available()}")
#mps_test = [torch.backends.mps.is_built(), torch.backends.mps.is_available()]
#mps_test_name = ["built", "available"]

#for n, t in zip(mps_test_name, mps_test):
#    print("mps is", n, ":", t)

"""
Note to self: mps issues is fixed by updating 
OS on mac. 

Also - functionality depends on specifying model device as below
"""
train_losses = []
test_losses = []

model = model.to(device)
with torch.no_grad(): 
    #since we're not training, gradients are disabled
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("training loss", train_loss)
print("test loss/val loss", val_loss)
# it works! 

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
num_epochs = 3
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, 
    num_epochs = num_epochs, eval_freq = 5, eval_iter = 5,
    start_context = "Every effort moves you ", tokenizer = tokenizer)


def plot_eval(epochs_seen, tokens_seen, train_losses, val_losses, train_label, val_label, y_label, save_as):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label = train_label)
    ax1.plot(epochs_seen, val_losses, linestyle = "-.", label = val_label)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(y_label)
    ax1.legend(loc = "upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha = 0)
    fig.tight_layout()
    plt.savefig(save_as, bbox_inches = "tight")
    plt.show()

train_losses = numpy.array(train_losses)
test_losses = numpy.array(test_losses)
#epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
#plot_eval(epochs_tensor, tokens_seen, train_losses, val_losses, "Training loss", "Validation loss", "Loss", save_as = "losses_pretraining_test.pdf")
#plot_eval(epochs_tensor, tokens_seen, numpy.exp(train_losses), numpy.exp(val_losses), "Training perplexity", "Validation perplexity" ,"Perplexity", save_as = "perplexity_pretraining_test.pdf")

generate(model, text_to_token_ids("Every effort moves you", tokenizer), 
         max_new_tokens = 15, 
         context_size = GPT_CONFIG_124M["context_length"],
         top_k = 25,
         temperature = 1.4)



