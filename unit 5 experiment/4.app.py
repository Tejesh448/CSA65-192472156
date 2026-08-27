import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Translation and Paraphrasing")

st.title("Text Translation and Paraphrasing Using Local LLM")

st.write("Perform translation and paraphrasing using a locally running LLM.")

@st.cache_resource
def load_model():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()


operation = st.selectbox(
    "Select Operation",
    ["Translation", "Paraphrasing"]
)


if operation == "Translation":

    language = st.selectbox(
        "Select Target Language",
        ["French", "German", "Spanish", "Hindi", "Tamil"]
    )

    text = st.text_area(
        "Enter text to translate:",
        placeholder="Example: Artificial Intelligence is useful."
    )

else:

    text = st.text_area(
        "Enter text to paraphrase:",
        placeholder="Example: Artificial Intelligence is changing the world."
    )


if st.button("Process Text"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        with st.spinner("Processing..."):

            if operation == "Translation":

                prompt = (
                    "Translate the following English text to "
                    + language
                    + ": "
                    + text
                )

            else:

                prompt = (
                    "Paraphrase the following text in simple English: "
                    + text
                )


            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )


            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=4
                )


            result = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )


            if operation == "Translation":
                st.subheader("Translated Text")
            else:
                st.subheader("Paraphrased Text")


            st.write(result)