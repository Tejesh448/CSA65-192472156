import gradio as gr
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from sklearn.feature_extraction.text import TfidfVectorizer


print("Loading AI Research Assistant model...")

# ---------------------------------------
# Load Pre-trained AI Model
# ---------------------------------------

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("AI Research Assistant model loaded successfully!")


# ---------------------------------------
# AI Text Generation Function
# ---------------------------------------

def generate_ai_text(prompt, max_tokens=180):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            num_beams=5,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return result.strip()


# ---------------------------------------
# Keyword Extraction
# ---------------------------------------

def extract_keywords(topic):

    try:

        words = topic.lower().split()

        # Remove common words
        stop_words = {
            "the",
            "and",
            "in",
            "of",
            "for",
            "to",
            "a",
            "an",
            "on",
            "with",
            "using",
            "from"
        }

        keywords = []

        for word in words:

            word = word.strip(
                ".,!?;:()[]{}"
            )

            if (
                word
                and word not in stop_words
                and len(word) > 2
            ):

                keywords.append(word)

        # Remove duplicates
        keywords = list(dict.fromkeys(keywords))

        if not keywords:

            return "No keywords found."

        return ", ".join(keywords[:10])

    except Exception:

        return "No keywords found."


# ---------------------------------------
# Generate Relevant Information
# ---------------------------------------

def generate_information(topic):

    prompt = f"""
Explain the research topic "{topic}".

Provide useful information for an engineering student.

Include:
1. Introduction
2. Important concepts
3. Applications
4. Advantages
5. Challenges

Give a clear answer using short paragraphs.
"""

    result = generate_ai_text(
        prompt,
        220
    )

    # Check whether model produced useful output
    if (
        not result
        or result.lower() == topic.lower()
        or len(result.split()) < 15
    ):

        result = f"""
Introduction:
{topic} is an important area of study in modern engineering and technology.
It involves the application of technical concepts and computational methods
to solve practical problems.

Important Concepts:
The main concepts depend on the specific application of {topic}.
Researchers study its methods, technologies, performance, and implementation.

Applications:
{topic} can be applied in engineering systems, research projects,
automation, data analysis, and real-world technological applications.

Advantages:
It can improve efficiency, accuracy, productivity, and decision-making
when properly designed and implemented.

Challenges:
Important challenges include cost, computational requirements,
data quality, security, reliability, and practical implementation.
"""

    return result.strip()


# ---------------------------------------
# Generate Concise Summary
# ---------------------------------------

def generate_summary(topic):

    prompt = f"""
Summarize "{topic}" for an engineering student.

Write 4 short and meaningful sentences.
Explain what it is, why it is important,
where it is used, and one major challenge.
"""

    result = generate_ai_text(
        prompt,
        100
    )

    # Fallback if model output is too short
    if (
        not result
        or result.lower() == topic.lower()
        or len(result.split()) < 10
    ):

        result = (
            f"{topic} is an important engineering and technology topic "
            f"with applications in modern technical systems. "
            f"It can help improve efficiency, automation, and problem solving "
            f"when appropriate methods are used. "
            f"The topic is useful for research and practical engineering applications. "
            f"Important challenges include implementation cost, reliability, "
            f"data requirements, and system performance."
        )

    return result.strip()


# ---------------------------------------
# Main Research Assistant
# ---------------------------------------

def research_assistant(topic):

    if topic is None or not topic.strip():

        return (
            "Please enter a research topic.",
            "",
            ""
        )

    topic = topic.strip()

    try:

        # Generate information
        information = generate_information(
            topic
        )

        # Extract keywords
        keywords = extract_keywords(
            topic
        )

        # Generate summary
        summary = generate_summary(
            topic
        )

        return (
            information,
            keywords,
            summary
        )

    except Exception as e:

        return (
            "Error: " + str(e),
            "",
            ""
        )


# ---------------------------------------
# Gradio Interface
# ---------------------------------------

interface = gr.Interface(

    fn=research_assistant,

    inputs=gr.Textbox(
        label="Enter Research Topic",
        placeholder="Example: Artificial Intelligence in Healthcare",
        lines=3
    ),

    outputs=[

        gr.Textbox(
            label="Relevant Information",
            lines=15
        ),

        gr.Textbox(
            label="Important Keywords",
            lines=4
        ),

        gr.Textbox(
            label="Concise Summary",
            lines=8
        )
    ],

    title="AI-Based Research Assistance System",

    description=(
        "Enter a research topic to generate relevant information, "
        "important keywords, and a concise summary using a "
        "pre-trained AI language model."
    )
)


# ---------------------------------------
# Start Application
# ---------------------------------------

if __name__ == "__main__":

    interface.launch(share=True)