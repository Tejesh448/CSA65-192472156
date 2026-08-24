import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader


# -----------------------------------
# Translation Models
# -----------------------------------

MODELS = {

    "Hindi": "Helsinki-NLP/opus-mt-en-hi",

    "Tamil": "Helsinki-NLP/opus-mt-en-dra",

    "Telugu": "Helsinki-NLP/opus-mt-en-dra",

    "Kannada": "Helsinki-NLP/opus-mt-en-dra",

    "Malayalam": "Helsinki-NLP/opus-mt-en-dra",

    "Bengali": "Helsinki-NLP/opus-mt-en-inc",

    "Marathi": "Helsinki-NLP/opus-mt-en-inc",

    "Gujarati": "Helsinki-NLP/opus-mt-en-inc",

    "Punjabi": "Helsinki-NLP/opus-mt-en-inc"
}


# -----------------------------------
# Load Model
# -----------------------------------

def load_translation_model(language):

    model_name = MODELS[language]

    print("Loading model for:", language)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# -----------------------------------
# Extract PDF Text
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
# Translation Function
# -----------------------------------

def translate_document(pdf_file, language):

    if pdf_file is None:

        return "Please upload an English engineering PDF document."

    if language is None:

        return "Please select a target language."

    try:

        print("Selected language:", language)

        text = extract_pdf_text(pdf_file)

        if not text.strip():

            return "The PDF does not contain readable text."

        # Limit text length
        text = text[:5000]

        tokenizer, model = load_translation_model(language)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        translated_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=4
        )

        translated_text = tokenizer.decode(
            translated_ids[0],
            skip_special_tokens=True
        )

        return translated_text

    except Exception as e:

        return "Error: " + str(e)


# -----------------------------------
# Gradio Interface
# -----------------------------------

interface = gr.Interface(

    fn=translate_document,

    inputs=[

        gr.File(
            label="Upload English Engineering PDF",
            file_types=[".pdf"],
            type="filepath"
        ),

        gr.Dropdown(

            choices=[
                "Hindi",
                "Tamil",
                "Telugu",
                "Kannada",
                "Malayalam",
                "Bengali",
                "Marathi",
                "Gujarati",
                "Punjabi"
            ],

            label="Select Target Indian Language",

            value="Hindi"
        )
    ],

    outputs=gr.Textbox(
        label="Translated Engineering Document",
        lines=15
    ),

    title="AI Engineering Document Translator",

    description=(
        "Upload an English engineering PDF and "
        "select an Indian language. The AI model "
        "will translate the document into the "
        "selected language."
    )
)


# -----------------------------------
# Start Application
# -----------------------------------

if __name__ == "__main__":

    interface.launch()