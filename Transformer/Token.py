

"""" 
1. GPT-2 use Byte Pair Encoding tokenizer for converting text to token
This has the implementation of converting text into token

"""

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Getting the raw-text
with open("the-verdict.txt", 'r', encoding='utf-8') as f:
    raw_text = f.read()  # Total raw_text length is about of 20480

# Encoding text
enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[:50]


""" 
LLMs generate text in accordance with sliding window
An example of it given below for both embedding and token scale
"""
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

# Sliding window on embedding representation
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "--->", desired)

"""
[198] ---> 40
[198, 40] ---> 367
[198, 40, 367] ---> 2885
[198, 40, 367, 2885] ---> 1464
"""

# Sliding window on tokens
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))

""" 
---> I

I --->  H

I H ---> AD

I HAD --->  always
"""
