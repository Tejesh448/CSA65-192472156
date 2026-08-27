import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Local Text Generator")

st.title("Local LLM Text Generator")
st.write("Generate text using a locally running Hugging Face model.")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2"
    )

generator = load_model()

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Example: Artificial Intelligence is"
)

if st.button("Generate Text"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):

            result = generator(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )

            st.subheader("Generated Text")
            st.write(result[0]["generated_text"])