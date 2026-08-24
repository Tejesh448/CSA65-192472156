from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning"
]

for p in prompts:
    print("Prompt:", p)
    result = generator(p, max_length=30)
    print(result[0]["generated_text"])
    print("-" * 40)
