import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from engineering_data import engineering_info


# -----------------------------------
# Load Pre-trained AI Model
# -----------------------------------

print("Loading Engineering Support AI Model...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

print("Engineering Support AI Model loaded successfully!")


# -----------------------------------
# Engineering Support Chatbot
# -----------------------------------

def engineering_chatbot(question):

    question = question.strip()

    # Empty input
    if not question:
        return "Please enter a technical question."

    q = question.lower()


    # -----------------------------------
    # Python
    # -----------------------------------

    if "what is python" in q or "define python" in q:

        return (
            "Python is a high-level programming language known for its "
            "simple syntax. It is widely used in web development, data "
            "science, artificial intelligence, machine learning and automation."
        )


    # -----------------------------------
    # Machine Learning
    # -----------------------------------

    if (
        "what is machine learning" in q
        or "define machine learning" in q
    ):

        return (
            "Machine learning is a branch of artificial intelligence that "
            "allows computers to learn patterns from data and make predictions "
            "or decisions without being explicitly programmed for every task."
        )


    # -----------------------------------
    # Supervised Learning
    # -----------------------------------

    if "supervised learning" in q:

        return (
            "Supervised learning uses labeled training data to learn a "
            "relationship between inputs and outputs. Classification and "
            "regression are common examples."
        )


    # -----------------------------------
    # Unsupervised Learning
    # -----------------------------------

    if "unsupervised learning" in q:

        return (
            "Unsupervised learning works with unlabeled data. It identifies "
            "patterns or groups in the data. Clustering is a common example."
        )


    # -----------------------------------
    # NLP
    # -----------------------------------

    if (
        "what is nlp" in q
        or "what is natural language processing" in q
        or "define nlp" in q
    ):

        return (
            "Natural Language Processing (NLP) is a branch of AI that enables "
            "computers to understand, process and generate human language."
        )


    # -----------------------------------
    # Database
    # -----------------------------------

    if "what is a database" in q or "what is database" in q:

        return (
            "A database is an organized collection of data that can be "
            "stored, managed, searched and retrieved efficiently."
        )


    # -----------------------------------
    # SQL
    # -----------------------------------

    if "what is sql" in q or "define sql" in q:

        return (
            "SQL stands for Structured Query Language. It is used to create, "
            "read, update and delete data in relational databases."
        )


    # -----------------------------------
    # Stack
    # -----------------------------------

    if "stack" in q:

        return (
            "A stack is a linear data structure that follows the LIFO "
            "principle, meaning Last In First Out. Push and pop are common "
            "stack operations."
        )


    # -----------------------------------
    # Queue
    # -----------------------------------

    if "queue" in q:

        return (
            "A queue is a linear data structure that follows the FIFO "
            "principle, meaning First In First Out. Enqueue and dequeue "
            "are common queue operations."
        )


    # -----------------------------------
    # IP Address
    # -----------------------------------

    if (
        "ip address" in q
        or "what is ip" in q
    ):

        return (
            "An IP address is a numerical address used to identify a device "
            "on a computer network and enable communication between devices."
        )


    # -----------------------------------
    # Operating System
    # -----------------------------------

    if (
        "operating system" in q
        or "what is os" in q
    ):

        return (
            "An operating system is system software that manages computer "
            "hardware, software resources and provides services for programs. "
            "Examples include Windows and Linux."
        )


    # -----------------------------------
    # Neural Network
    # -----------------------------------

    if "neural network" in q:

        return (
            "A neural network is a machine learning model inspired by the "
            "human brain. It contains interconnected nodes arranged in "
            "input, hidden and output layers."
        )


    # -----------------------------------
    # Algorithm
    # -----------------------------------

    if "algorithm" in q:

        return (
            "An algorithm is a finite sequence of clear, step-by-step "
            "instructions used to solve a problem or perform a task."
        )


    # -----------------------------------
    # Debugging
    # -----------------------------------

    if "debugging" in q or "debug" in q:

        return (
            "Debugging is the process of finding, analyzing and fixing "
            "errors or bugs in a computer program."
        )


    # -----------------------------------
    # ModuleNotFoundError
    # -----------------------------------

    if "modulenotfounderror" in q:

        return (
            "ModuleNotFoundError occurs when Python cannot find a required "
            "module. Check the module name, install the required package "
            "using pip install package_name, and make sure the correct "
            "virtual environment is activated."
        )


    # -----------------------------------
    # SyntaxError
    # -----------------------------------

    if "syntaxerror" in q:

        return (
            "SyntaxError occurs when Python code does not follow the correct "
            "syntax. Check brackets, quotation marks, indentation and "
            "Python keywords."
        )


    # -----------------------------------
    # Cybersecurity
    # -----------------------------------

    if "cybersecurity" in q or "cyber security" in q:

        return (
            "Cybersecurity is the practice of protecting computers, networks, "
            "applications and data from unauthorized access, attacks and "
            "other security threats."
        )


    # -----------------------------------
    # Data Structures
    # -----------------------------------

    if "data structure" in q or "data structures" in q:

        return (
            "Data structures are methods used to organize and store data "
            "efficiently. Examples include arrays, linked lists, stacks, "
            "queues, trees and graphs."
        )


    # -----------------------------------
    # Programming
    # -----------------------------------

    if "programming" in q:

        return (
            "Programming is the process of writing instructions that a "
            "computer can execute to perform a specific task."
        )


    # -----------------------------------
    # Technical Keywords
    # -----------------------------------

    technical_keywords = [
        "python",
        "programming",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "ai",
        "nlp",
        "database",
        "sql",
        "network",
        "networking",
        "ip",
        "operating system",
        "data structure",
        "algorithm",
        "debug",
        "error",
        "computer",
        "software",
        "hardware",
        "neural network",
        "cybersecurity"
    ]


    # -----------------------------------
    # Reject Non-Technical Questions
    # -----------------------------------

    if not any(word in q for word in technical_keywords):

        return (
            "Sorry, I can only answer engineering and technical questions "
            "related to programming, AI, machine learning, NLP, databases, "
            "networking and computer science."
        )


    # -----------------------------------
    # AI Model Prompt
    # -----------------------------------

    prompt = f"""
You are an engineering technical support chatbot.

Use the following engineering knowledge to answer the question.

Engineering Knowledge:
{engineering_info}

Technical Question:
{question}

Give a clear, short and useful answer.
If the question describes an error or technical problem, provide a solution.
Do not repeat the question.
Do not repeat the entire knowledge base.
"""


    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )


    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )


    # Decode
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    return answer


# -----------------------------------
# Gradio Interface
# -----------------------------------

interface = gr.Interface(

    fn=engineering_chatbot,

    inputs=gr.Textbox(
        label="Technical Question",
        placeholder="Ask an engineering or technical question..."
    ),

    outputs=gr.Textbox(
        label="Engineering Support Answer"
    ),

    title="Engineering Support AI Chatbot",

    description=(
        "Ask technical questions about Python, AI, Machine Learning, "
        "NLP, databases, networking, programming and computer science."
    )
)


# -----------------------------------
# Start Application
# -----------------------------------

if __name__ == "__main__":
    interface.launch()