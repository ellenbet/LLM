from utils import TokenizerV1

import os 
import urllib.request
import re

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r", encoding = "utf-8") as f:
        raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
words = sorted(set(preprocessed))
vocab = {token:integer for integer, token in enumerate(words)}
print("Vocabulary size: ", len(words))

# Testing tokenizer

tokenizer = TokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)