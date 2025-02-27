import torch
from utils import DataLoader, Dataset, create_dataloader_v1
import os 
import urllib.request
import re
import importlib
print("tiktoken version:", importlib.metadata.version("tiktoken"))

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

try: 
    with open("the-verdict.txt", "r", encoding = "utf-8") as f:
        raw_text = f.read()
except FileNotFoundError:
    print("Not able to fetch file online or from path.")

"""
We want to create an embedding matrix with n rows and m columns.
n is the higest token id + 1
m is the output dimension we want

We compare this with the nn.Linear layer, which has similar functionalities.

"""

token_ids = torch.tensor([2, 3, 1])
n = max(token_ids) + 1
m = 5

embedding = torch.nn.Embedding(n, m)
print(" Embedding weights:\n", embedding.weight, "\ndim:", embedding.weight.shape)

id = 1
print(f"\nObtain vector representation of training example with token id = {id}")
print(embedding(torch.tensor([id])))
print(f"This is essentially retrieving row {id} of the n = {n} row matrix.\n")

# This is essentially the same as using nn.Linear()
onehot = torch.nn.functional.one_hot(token_ids)
print("One-hot encoding of token ids:\n", onehot)

# Initialize a linear layer: 
linear = torch.nn.Linear(n, m, bias = False)
print("\nLinear weights:\n", linear.weight, "\ndim:", linear.weight.shape)

# Adding the embedding weights to the linear layer
linear.weight = torch.nn.Parameter(embedding.weight.T)
print("\nUsing one-hot encoding of token ids to achieve same vector representation as above: \n", linear(onehot.float()))
print("\nThis gives the exact same result as inputing token IDs into the embedding object:\n", embedding(token_ids))
print("\n\nConclusion: since one-hot encoding requires a lot of multiplications with zero, we perfer embeddings!")


"""
Creating token embeddings for the LLM
"It can be seen as a neural network layer that can be optimized via backpropagation"

Embeddings contain embedding layer weight matrices. The layer is converting token IDs into identical vector
representations regardless of where they are in the input sequence. 

We combine the token embeddings discussed here, with positional embeddings. This is the input embedding
we use for the LLM. 

"""

# BP encoder has vocab size as specified below
vocab_size = 50257
m = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, m)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size = 8, max_length = max_length, stride = max_length, shuffle = False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("\nToken IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("\nToken embeddings, shape:", token_embeddings.shape)

# Another embedding for positions
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, m)
print("\n Positional embedding weights:\n", pos_embedding_layer.weight)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("\nPositional embedding shape:", pos_embeddings.shape)

# Input embedding for LLM is the sum of token and positional embeddings
input_embedding = token_embeddings + pos_embeddings
print("\nInput embeddings:\n", input_embedding)
print("\nInput embedding shape:", input_embedding.shape)

# Keep in mind that number of matrices (or cube depth) comes first
print("\n\nTOKEN EMBEDDINGS:\n", token_embeddings, "\nshape:", token_embeddings.shape)
print("\n\nPOSSITIONAL EMBEDDINGS:\n", pos_embeddings, "\nshape:", pos_embeddings.shape)