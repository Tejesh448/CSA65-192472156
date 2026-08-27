import os
import ollama
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


print("==============================================")
print(" Local RAG Engineering Troubleshooting System")
print("==============================================")


# -----------------------------------------
# 1. Load embedding model
# -----------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# -----------------------------------------
# 2. Create vector database
# -----------------------------------------

client = chromadb.PersistentClient(
    path="./troubleshooting_db"
)

collection = client.get_or_create_collection(
    name="engineering_troubleshooting"
)


# -----------------------------------------
# 3. Read technical PDF documents
# -----------------------------------------

folder = "engineering_docs"

chunks = []
ids = []

for filename in os.listdir(folder):

    if filename.lower().endswith(".pdf"):

        path = os.path.join(folder, filename)

        print("\nReading:", filename)

        reader = PdfReader(path)

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                words = text.split()

                # Divide document into chunks
                for i in range(0, len(words), 150):

                    chunk = " ".join(
                        words[i:i + 150]
                    )

                    if chunk.strip():

                        chunks.append(chunk)

                        ids.append(
                            f"{filename}_page_{page_number}_chunk_{i}"
                        )


# -----------------------------------------
# 4. Check documents
# -----------------------------------------

if len(chunks) == 0:

    print("\nNo technical PDF found.")

    print(
        "Place a PDF inside the engineering_docs folder."
    )

    exit()


print("\nTotal document chunks:", len(chunks))


# -----------------------------------------
# 5. Create embeddings
# -----------------------------------------

print("\nCreating document embeddings...")

embeddings = embedding_model.encode(
    chunks
).tolist()


# -----------------------------------------
# 6. Store documents in ChromaDB
# -----------------------------------------

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("Documents stored in vector database.")


# -----------------------------------------
# 7. Get troubleshooting problem
# -----------------------------------------

problem = input(
    "\nDescribe your engineering problem: "
)


if problem.strip() == "":

    print(
        "Please describe an engineering problem."
    )

    exit()


# -----------------------------------------
# 8. Convert problem into embedding
# -----------------------------------------

problem_embedding = embedding_model.encode(
    [problem]
).tolist()[0]


# -----------------------------------------
# 9. Retrieve relevant information
# -----------------------------------------

results = collection.query(
    query_embeddings=[problem_embedding],
    n_results=3
)


relevant_documents = results["documents"][0]


context = "\n\n".join(
    relevant_documents
)


# -----------------------------------------
# 10. Create troubleshooting prompt
# -----------------------------------------

prompt = f"""
You are an engineering troubleshooting assistant.

Use ONLY the technical information provided in the context.

ENGINEERING DOCUMENT CONTEXT:
{context}

USER PROBLEM:
{problem}

Instructions:

1. Identify the likely problem.
2. Explain the possible cause.
3. Provide step-by-step troubleshooting recommendations.
4. Keep the recommendations practical and clear.
5. Do not invent technical information.
6. If the required information is not present in the
   documents, clearly state:
   "The required information is not available in the
   technical documents."

Format your response as:

Problem:
Cause:
Troubleshooting Steps:
1.
2.
3.
4.

Final Recommendation:
"""


# -----------------------------------------
# 11. Generate answer using Ollama
# -----------------------------------------

print("\nGenerating troubleshooting recommendations...")

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


# -----------------------------------------
# 12. Display result
# -----------------------------------------

print("\n==============================================")
print(" TROUBLESHOOTING RECOMMENDATION")
print("==============================================")

print(answer)


print("\n==============================================")
print(" Retrieved Information")
print("==============================================")

for i, document in enumerate(
    relevant_documents,
    start=1
):

    print(f"\nSource {i}:")
    print(document[:500])