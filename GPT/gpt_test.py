import tiktoken
import torch
from gpt import *

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

"""
Tokenizer tests
"""

batch.append(torch.tensor(tokenizer.encode(txt1)))
print("batch before stack: \n", batch)
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim = 0)
print("batch after stack: \n", batch)

"""
GPT tests

"""

# Dummy models removed


"""
FFNN tests

"""

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print("FFN out shape:\n", out.shape)

"""
Shortcut connections tests

Using shortcuts prevents gradients from disapearing in 
the early layers. 

"""
layer_sizes = [3, 3, 3, 3, 3, 1]  
sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print()
print_gradients(model_with_shortcut, sample_input)



"""
Combining the previous components into a transformer block

- multi-head attention module
- linear layers
- ffnn
- dropout and shortcut connections

"""
    
torch.manual_seed(123)

x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("\nInput shape:", x.shape)
print("Output shape:", output.shape)


"""
Coding the actual GPT model
Which consists of: 
- a GPT backbone
- layer normalization modules
- GELU activation modules
- FFNNs
- shortcut connections

All of which are combined into a transformer block, which again is 
implemented several times into the full GPT model. 

"""
    

# And we test! 
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# Print out number of parameters: 

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")


# Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4

# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)

print(f"Total size of the model: {total_size_mb:.2f} MB") # emb_dim = 768 is the small GPT2



