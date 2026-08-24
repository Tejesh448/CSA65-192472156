from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

text = "The movie was wonderful."

print(sentiment(text))
