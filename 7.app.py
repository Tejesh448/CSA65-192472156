import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import torch


print("Loading AI summarization model...")


# -----------------------------------
# Load Pre-trained Model
# -----------------------------------

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("AI summarization model loaded successfully!")


# -----------------------------------
# Extract Text from PDF
# -----------------------------------

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------------
# Summarization Function
# -----------------------------------

def summarize_document(pdf_file):

    if pdf_file is None:

        return "Please upload an engineering PDF document."

    try:

        text = extract_pdf_text(pdf_file)

        if not text.strip():

            return "The PDF does not contain readable text."

        # Limit document length
        text = text[:10000]

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        # Generate summary
        with torch.no_grad():

            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                min_length=40,
                num_beams=4,
                early_stopping=True
            )

        # Convert generated tokens to text
        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary

    except Exception as e:

        return "Error: " + str(e)


# -----------------------------------
# Gradio Interface
# -----------------------------------

interface = gr.Interface(

    fn=summarize_document,

    inputs=gr.File(
        label="Upload Engineering PDF",
        file_types=[".pdf"],
        type="filepath"
    ),

    outputs=gr.Textbox(
        label="Generated Summary",
        lines=10
    ),

    title="AI Engineering Document Summarizer",

    description=(
        "Upload an engineering PDF document "
        "and the pre-trained BART AI model "
        "will generate a short and meaningful summary."
    )
)


# -----------------------------------
# Start Application
# -----------------------------------

if __name__ == "__main__":

    interface.launch()