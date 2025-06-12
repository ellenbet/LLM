
from torch.utils.data import DataLoader
from utils import InstructionDataset, custom_collate_fn, token_ids_to_text, text_to_token_ids, generate, format_input, plot_eval, train_model_simple, set_plt_params
import json
import time
import torch
from functools import partial
import tiktoken
from gpt_download import load_gpt2_params_from_tf_ckpt
from gpt import GPTModel, load_weights_into_gpt, GPT_CONFIG_124M
import tensorflow as tf
import os
import numpy as np
import re 

set_plt_params()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

file_path = "/Users/ellen-beatetysvaer/Documents/V24/FYS5429/LLM/testing-scripts/instruction-data.json"

with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

"""
# Testing that it looks as it should
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

print(inputs[0])

print(targets[0])
"""

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

gpt = GPTModel(BASE_CONFIG)
gpt.eval()

# Load settings and params
model_dir = "gpt2/355M"
tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
gpt.to(device)
print("done loading!\ntesting...")
load_weights_into_gpt(gpt, params)
gpt.eval()
"""

# How to make lasagna?

torch.manual_seed(123)
token_ids = generate(
    model = gpt, 
    idx = text_to_token_ids("How to make lasagna:",  tokenizer).to(device),
    max_new_tokens = 250, 
    context_size = BASE_CONFIG["context_length"],
    top_k = 100, 
    temperature = 0.9
)

print("output:\n", token_ids_to_text(token_ids, tokenizer))
"""

input_text = format_input(val_data[0])
token_ids = generate(
    model = gpt ,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

response_text = (
    generated_text[len(input_text):]
    .replace("### Response:", "")
    .strip()
)
print(response_text)
start_time = time.time()

torch.manual_seed(123)



# on to some actual finetuning training!! 

optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    gpt, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# visualizing loss: 

train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_eval(epochs_tensor, tokens_seen, train_losses, val_losses, train_label = "Train loss", val_label = "Validation loss", y_label = "Loss", save_as = "fine-tuning_loss.pdf")
plot_eval(epochs_tensor, tokens_seen, np.exp(train_losses), np.exp(val_losses), train_label = "Train perplexity", val_label = "Validation perplexity", y_label = "Perplexity", save_as = "fine-tuning_perplexity.pdf")

# consider https://www.gradio.app/

file_name = f"fine-tuned_1206_{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(gpt.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

