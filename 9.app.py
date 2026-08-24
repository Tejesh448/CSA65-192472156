import gradio as gr
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# Extract Text from PDF
# -----------------------------------

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


# -----------------------------------
# Resume Screening Function
# -----------------------------------

def screen_resumes(resume_files, job_description):

    if not resume_files:

        return "Please upload at least one resume.", None

    if not job_description.strip():

        return "Please enter a job description.", None

    candidates = []

    # Extract text from all resumes
    for resume in resume_files:

        try:

            text = extract_pdf_text(resume)

            if not text.strip():

                text = "No readable text found in resume."

            candidates.append({
                "Resume": resume.split("\\")[-1],
                "Text": text
            })

        except Exception as e:

            candidates.append({
                "Resume": resume.split("\\")[-1],
                "Text": ""
            })


    # Prepare documents
    documents = [job_description]

    for candidate in candidates:

        documents.append(candidate["Text"])


    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(documents)


    # Calculate similarity
    job_vector = vectors[0]

    results = []

    for i, candidate in enumerate(candidates):

        resume_vector = vectors[i + 1]

        similarity = cosine_similarity(
            job_vector,
            resume_vector
        )[0][0]

        score = round(similarity * 100, 2)

        results.append({
            "Candidate": candidate["Resume"],
            "Match Score (%)": score
        })


    # Sort candidates by score
    results = sorted(
        results,
        key=lambda x: x["Match Score (%)"],
        reverse=True
    )


    # Add ranking
    for rank, result in enumerate(results, start=1):

        result["Rank"] = rank


    # Create DataFrame
    dataframe = pd.DataFrame(results)

    dataframe = dataframe[
        ["Rank", "Candidate", "Match Score (%)"]
    ]


    # Create text result
    result_text = "Resume Screening Completed Successfully!\n\n"

    for result in results:

        result_text += (
            f"Rank {result['Rank']}: "
            f"{result['Candidate']} - "
            f"{result['Match Score (%)']}% Match\n"
        )


    return result_text, dataframe


# -----------------------------------
# Gradio Interface
# -----------------------------------

interface = gr.Interface(

    fn=screen_resumes,

    inputs=[

        gr.File(
            label="Upload Candidate Resumes (PDF)",
            file_types=[".pdf"],
            file_count="multiple",
            type="filepath"
        ),

        gr.Textbox(
            label="Engineering Job Description",
            placeholder=(
                "Example: We are looking for a Python Engineer "
                "with knowledge of Machine Learning, SQL, "
                "Data Analysis and NLP."
            ),
            lines=8
        )
    ],

    outputs=[

        gr.Textbox(
            label="Screening Result",
            lines=10
        ),

        gr.Dataframe(
            label="Candidate Ranking",
            headers=[
                "Rank",
                "Candidate",
                "Match Score (%)"
            ]
        )
    ],

    title="AI-Based Engineering Resume Screening System",

    description=(
        "Upload multiple engineering resumes and enter "
        "a job description. The system uses NLP-based "
        "TF-IDF and Cosine Similarity to rank candidates "
        "according to their job-description match."
    )
)


# -----------------------------------
# Start Application
# -----------------------------------

if __name__ == "__main__":

    interface.launch()