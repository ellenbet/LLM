import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Testing how BPE encodes unknown words in leu of 
# just replacing with <unk>
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

ints = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ints)

# BPE tokenizers break down unknown words into subwords and individual characters
strs = tokenizer.decode(ints)
print(strs)