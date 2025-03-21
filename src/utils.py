
# regular expression operations re
import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


# Tokenizer from Raschka's LLM book
class Tokenizer:
    """
    Class creates an instance that takes in a vocabulary (from a set {} object) 
    and returns encoded or decoded text based on existing vocabulary.

    Methods: 
    encode() takes in text and translates texts to 
    integer sequences based on existing vocabulary
    - "encoder turns text into token IDs"

    decode() takes in integer sequences and returns 
    text with correct spacing. 
    - "decoder turns token IDs back into text"
    
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preproc = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preproc = [
            item.strip() for item in preproc if item.strip()
        ]

        # adding unk if the item is not in vocab
        preproc = [
            item if item in self.str_to_int else "<|unk|>" for item in preproc
        ]
        #print(preproc) -> debugging after issues related to jupyter
        ids = [self.str_to_int[s] for s in preproc]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize text: 
        token_ids = tokenizer.encode(txt, allowed_special = {"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt,
                         tokenizer = None,
                         batch_size = 4, 
                         max_length = 256, 
                         stride = 128, 
                         shuffle = True,
                         drop_last = True,
                         num_workers = 0):
    if tokenizer == None: 
        tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size : ]

        with torch.no_grad():
            logits = model(idx_cond)
        
        #( batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(probas, dim = -1, keepdim = True)
        idx = torch.cat((idx, idx_next), dim = 1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #batch dimension/matrix created
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) #remove batch dimension/make flat
    return tokenizer.decode(flat.tolist())


# Loss functions
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches == None: 
        num_batches = len(data_loader)
    else: 
        # in case of non-consistenct between loader and batch size
        num_batches = min(num_batches, len(data_loader))
    for input_batch, target_batch in data_loader:
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train() # set to training model

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() #reset pr. batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() #get loss gradient
            optimizer.step() #update model weights using loss gradients
            tokens_seen += input_batch.numel() #numel = number of elements
            global_step += 1
        
            # evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch+1}\nStep: {global_step:06d}\nTrain loss: {train_loss:.3f}\nValidation loss: {val_loss:.3f}")
            
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model, idx = encoded, 
            max_new_tokens = 50, context_size = context_size )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()



    

        
