import ollama

print("==========================================")
print(" Local LLM Hallucination Demonstration")
print("==========================================")

# Reference information
reference = """
The Department of Computer Science was established in 2010.
It offers undergraduate and postgraduate programs in Computer Science.
The department has 25 faculty members and provides laboratories
for programming, artificial intelligence and data science.
The department library contains 1200 technical books.
"""

print("\nREFERENCE INFORMATION:")
print(reference)

question = input("\nEnter your question: ")

if question.strip() == "":
    print("Please enter a question.")
else:

    prompt = f"""
You are answering questions using ONLY the reference information below.

REFERENCE:
{reference}

QUESTION:
{question}

Rules:
- Do not invent information.
- If the answer is not available in the reference, say:
  "The information is not available in the reference."
- Give a short answer.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print("\nLLM ANSWER:")
    print(answer)

    print("\nREFERENCE ANALYSIS:")

    if "not available" in answer.lower():
        print("✓ No unsupported information was identified.")
    else:
        print("⚠ Check the answer against the reference information.")
        print("Possible hallucination may be present.")