from transformers import pipeline

qa = pipeline("question-answering")

context = """
Python is a popular programming language developed by Guido van Rossum.
"""

question = "Who developed Python?"

result = qa(question=question, context=context)

print(result)
