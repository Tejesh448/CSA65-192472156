import ollama

print("====================================")
print(" Local LLM Question Answering System")
print("====================================")

question = input("\nEnter your question: ")

if question.strip() == "":
    print("Please enter a question.")
else:

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    print("\nAnswer:")
    print(response["message"]["content"])