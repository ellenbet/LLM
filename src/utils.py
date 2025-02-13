
# regular expression operations re
import re



# Tokenizer from Raschka's LLM book
class TokenizerV1:
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
        #print(preproc) -> debugging after issues related to jupyter
        ids = [self.str_to_int[s] for s in preproc]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

        
