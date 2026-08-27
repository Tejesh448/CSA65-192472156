import os
import ollama
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Load embedding model
# --------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------
# 2. Create ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="./engineering_db"
)

collection = client.get_or_create_collection(
    name="engineering_documents"
)


# --------------------------------
# 3. Read PDF documents
# --------------------------------

documents_folder = "engineering_docs"

all_chunks = []
chunk_ids = []

for filename in os.listdir(documents_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(
            documents_folder,
            filename
        )

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                # Split text into small chunks
                words = text.split()

                for i in range(0, len(words), 100):

                    chunk = " ".join(
                        words[i:i + 100]
                    )

                    if chunk.strip():

                        all_chunks.append(chunk)

                        chunk_ids.append(
                            f"{filename}_{page_number}_{i}"
                        )


# --------------------------------
# 4. Create embeddings
# --------------------------------

print("Creating document embeddings...")

embeddings = embedding_model.encode(
    all_chunks
).tolist()


# --------------------------------
# 5. Store in ChromaDB
# --------------------------------

if len(all_chunks) > 0:

    collection.upsert(
        ids=chunk_ids,
        documents=all_chunks,
        embeddings=embeddings
    )

print("Documents stored in vector database.")


# --------------------------------
# 6. Ask a question
# --------------------------------

question = input(
    "\nEnter your technical question: "
)

if question.strip() == "":
    print("Please enter a question.")

else:

    # Create question embedding
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()[0]

    # Search vector database
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    relevant_chunks = results["documents"][0]

    context = "\n\n".join(
        relevant_chunks
    )


    # --------------------------------
    # 7. Create RAG prompt
    # --------------------------------

    prompt = f"""
You are an engineering technical assistant.

Answer the question using ONLY the provided context.

If the answer is not available in the context,
say "The information is not available in the document."

CONTEXT:
{context}

QUESTION:
{question}

Give a clear and concise technical answer.
"""


    # --------------------------------
    # 8. Send to Ollama
    # --------------------------------

    print("\nGenerating answer...")

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    answer = response[
        "message"
    ][
        "content"
    ]


    # --------------------------------
    # 9. Display answer
    # --------------------------------

    print("\n================================")
    print("RAG ANSWER")
    print("================================")

    print(answer)