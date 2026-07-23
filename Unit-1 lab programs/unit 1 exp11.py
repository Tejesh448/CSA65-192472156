from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = "Machine Learning is interesting."

output = tokenizer(sentence)

print(output)
