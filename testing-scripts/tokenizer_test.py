from utils import Tokenizer

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




preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
words = sorted(set(preprocessed))
words.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(words)}
print("Vocabulary size: ", len(words))

# Testing tokenizer

tokenizer = Tokenizer(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)
dec = tokenizer.decode(ids)
print(dec)

# Adding end-of-text etc special tokens

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))
print(text)

ids = tokenizer.encode(text)
dec = tokenizer.decode(ids)
print(dec)