
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

"""
--------------------------------------------
We compute attention scores by taking the dot product
between a selected query vector x_i and all other vectors
x_j in a matrix of token embeddings X, where n rows = number
of tokens/words and m columns = embedding dimension.
--------------------------------------------
"""

# We could compute the inner product directly: 
query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)

# Or through matrix multiplication: 
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# More commonly written as: 
attn_scores = inputs @ inputs.T
print(attn_scores)

# Finally, we normalize all rows so that they sum up
# to 1, using the softmax function: 
attn_weights = torch.softmax(attn_scores, dim = - 1)
print(attn_weights)


"""
--------------------------------------------
Self-attention with trainable weights:  nn 
--------------------------------------------

Also known as the "scaled dot-product attention".

Here we introduce weights that are updated during model 
training. Weights are essential to produce good context vectors. 

Wq, Wk and Wv
Query, key and valye weight matrices

Gives vectors q, k and v when multiplied with embedded input tokens.
In GPT models dim(q) == dim(x) (usually)
"""
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# requires_grad = False because of illustration purposes, in real models this is always True
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value


print("Query vector 2:", query_2)

keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
print("inputs.shape:", inputs.shape)

# dot project between query and key vector:

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("attention score 2.2", attn_score_22)


attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print("attention scores for query 2:", attn_scores_2)

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)


# All the information above makes out the SelfAttention_v1 class: 

import torch.nn as nn
torch.manual_seed(789)

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = 1)
        context_vec = attn_weights @ values
        return context_vec
    
sa_v1 = SelfAttention_v1(d_in, d_out)
print("First self attention forward pass", sa_v1(inputs))

# Using PyTorch Linear Layers: 

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=1)
        context_vec = attn_weights @ values
        return context_vec
    
sa_v2 = SelfAttention_v2(d_in, d_out)
print("Self attention forward pass with torch linear layers:", sa_v2(inputs))

"""
Hiding future words with causal attention:
Masking the attention weights above the diagonal ensures that future tokens
are not taken into consideration when calculating the context vectors with the
attention weight. 

We'll be converting the self-attention mechanism to a causal self-attention 
mechanism. Only previous positions are used to make model predictions. 

"""

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
print("attention weights:\n", attn_weights)

# Pytorch tril function to create a mask: 
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("Simple tril mask:\n", mask_simple)

masked_simple = attn_weights * mask_simple
print("Result of simple mask:\n", masked_simple)

"""
Potential issue: masking after softmax disrupts the softmax distrubution
We therefore re-normalize:

"""

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("Simple masked attention weights normalized:\n", masked_simple_norm)

# More efficient approach rather than tril: masking before applying the softmax

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# Dropout to reduce overfitting , dropout mask is applied with random positions to be dropped

dropout = nn.Dropout(0.5)
example = torch.ones(6,6)

print("Dropout example:\n", dropout(example))
print("Dropout on attention weights:\n", dropout(attn_weights))

"""
Implementing a compact causal self-attention class

"""

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[ : num_tokens, : num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context vectors:\n", context_vecs)
print("context vector shapes:\n", context_vecs.shape)






