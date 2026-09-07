
""" 
3. Implementation of converting token IDs into DataLoader for feeding into 
Transformer Model
"""

import tiktoken
from torch.utils.data import DataLoader
from GPTDataset import GPTDataset

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", 'r', encoding='utf-8') as f:
    raw_text = f.read()  # Getting raw text from tokenizer


def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text=txt, tokenizer=tokenizer,
                         max_length=max_length, stride=stride)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader


"""dataloader = create_dataloader(
    txt=raw_text, batch_size=1, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# Similarly
second_batch = next(data_iter)
print(second_batch)"""
