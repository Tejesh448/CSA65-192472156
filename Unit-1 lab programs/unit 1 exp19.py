from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of Artificial Intelligence"

result = generator(prompt, max_length=50)

print(result[0]["generated_text"])
