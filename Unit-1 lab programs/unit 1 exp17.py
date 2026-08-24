from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased")

sentence = "Machine Learning"

inputs = tokenizer(sentence, return_tensors="pt")

outputs = model(**inputs)

embedding = outputs.last_hidden_state

print(embedding.shape)
