import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Local QA System")

st.title("Question Answering Using Local LLM")

st.write("Ask a question about Artificial Intelligence.")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2"
    )

model = load_model()

knowledge = {
    "artificial intelligence":
        "Artificial Intelligence is a branch of computer science that enables machines to perform tasks that normally require human intelligence.",

    "machine learning":
        "Machine Learning is a method of teaching computers to learn patterns from data and make predictions or decisions.",

    "deep learning":
        "Deep Learning is a part of machine learning that uses neural networks with multiple layers to learn from large amounts of data.",

    "natural language processing":
        "Natural Language Processing is a field of AI that enables computers to understand and process human language.",

    "computer vision":
        "Computer Vision enables computers to understand and analyze images and videos.",

    "robotics":
        "Robotics is a field that combines AI, engineering and computer science to create machines that can perform tasks automatically."
}


question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is machine learning?"
)


if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        question_lower = question.lower()

        answer = None

        for topic, information in knowledge.items():

            if topic in question_lower:
                answer = information
                break

        if answer:

            st.subheader("Answer")
            st.write(answer)

        else:

            with st.spinner("Generating response..."):

                prompt = (
                    "Question: " + question +
                    "\nAnswer:"
                )

                result = model(
                    prompt,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7
                )

                generated = result[0]["generated_text"]

                answer = generated.replace(prompt, "").strip()

                st.subheader("Answer")

                if answer:
                    st.write(answer)
                else:
                    st.write(
                        "Sorry, I don't have information about this question."
                    )