import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Text Summarization")

st.title("Text Summarization Using Local LLM")

st.write("Enter text below and generate a short summary.")

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()


text = st.text_area(
    "Enter text to summarize:",
    height=250,
    placeholder="Enter a long paragraph here..."
)


if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    elif len(text.split()) < 30:
        st.warning("Please enter at least 30 words.")

    else:

        with st.spinner("Generating summary..."):

            prompt = "Summarize the following text: " + text

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    min_new_tokens=20,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            st.subheader("Summary")

            st.write(summary)