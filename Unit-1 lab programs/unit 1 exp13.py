from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

text = "Deep learning"

inputs = tokenizer.encode(text, return_tensors="pt")

output = model.generate(inputs, max_length=30)

print(tokenizer.decode(output[0]))
