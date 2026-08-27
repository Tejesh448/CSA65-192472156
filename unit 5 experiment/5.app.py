import ollama

print("Local LLM Text Generator")
print("------------------------")

prompt = input("Enter your prompt: ")

if prompt.strip() == "":
    print("Please enter a prompt.")
else:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nGenerated Text:")
    print(response["message"]["content"])