
from utils import DataLoader, Dataset, create_dataloader_v1
import os
import urllib

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


dataloader = create_dataloader_v1(raw_text, 
                                  batch_size = 1,
                                  max_length = 4, 
                                  stride = 1, )


print("Printing sliding windows of size 4 with token id's through example text.")
data_iter = iter(dataloader)
first_batch = next(data_iter)
print("First batch: ", first_batch)

second_batch = next(data_iter)
print("Second batch: ", second_batch)

print("\nPrinting sliding windows with no overlap and increased batch size")
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)