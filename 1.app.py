import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from college_data import college_info


# -----------------------------------
# Load Pre-trained AI Model
# -----------------------------------

print("Loading AI model...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

print("AI model loaded successfully!")


# -----------------------------------
# Chatbot Function
# -----------------------------------

def chatbot(question):

    # Remove unnecessary spaces
    question = question.strip()

    # -----------------------------------
    # Empty Input Handling
    # -----------------------------------

    if not question:
        return "Please enter a question."


    # Convert question to lowercase
    question_lower = question.lower()


    # -----------------------------------
    # Admission Questions
    # -----------------------------------

    admission_words = [
        "admission",
        "admit",
        "join",
        "joining",
        "enroll",
        "enrollment",
        "eligibility"
    ]

    if any(word in question_lower for word in admission_words):

        return (
            "Students can contact the admission office for information "
            "about the admission procedure, eligibility and required documents."
        )


    # -----------------------------------
    # Examination Questions
    # -----------------------------------

    exam_words = [
        "exam",
        "exams",
        "examination",
        "examinations",
        "semester exam",
        "semester examination",
        "test"
    ]

    if any(word in question_lower for word in exam_words):

        return (
            "Semester examinations are conducted according to the academic "
            "calendar. Students should check official college notifications "
            "for the examination dates."
        )


    # -----------------------------------
    # Direct College Information
    # -----------------------------------

    if "library" in question_lower:

        return (
            "The library is open from 8:30 AM to 6:00 PM, "
            "Monday to Saturday."
        )


    if "hostel" in question_lower:

        return (
            "Yes, hostel facilities are available for students. "
            "Students can contact the college administration for hostel-related information."
        )


    if "transport" in question_lower or "bus" in question_lower:

        return (
            "College bus transportation is available on selected routes. "
            "Students can contact the transport office for route and timing information."
        )


    if "course" in question_lower or "courses" in question_lower:

        return (
            "The courses offered are Computer Science Engineering, "
            "Information Technology, Electronics and Communication Engineering, "
            "Mechanical Engineering and Civil Engineering."
        )


    if "college timing" in question_lower or "college timings" in question_lower:

        return (
            "The college timings are Monday to Friday, "
            "9:00 AM to 4:30 PM."
        )


    if "principal" in question_lower:

        return (
            "The principal office is located in the administrative block."
        )


    if "department" in question_lower:

        return (
            "The college has departments including Computer Science Engineering, "
            "Information Technology, Electronics and Communication Engineering, "
            "Mechanical Engineering and Civil Engineering."
        )


    if "contact" in question_lower:

        return (
            "Students can contact the college administrative office "
            "for further information."
        )


    # -----------------------------------
    # College Keywords
    # -----------------------------------

    college_keywords = [
        "college",
        "student",
        "campus",
        "facility",
        "facilities",
        "engineering",
        "office",
        "academic",
        "department"
    ]


    # -----------------------------------
    # Reject Unrelated Questions
    # -----------------------------------

    if not any(word in question_lower for word in college_keywords):

        return (
            "Sorry, I can only answer questions related to "
            "the engineering college."
        )


    # -----------------------------------
    # AI Model Prompt
    # -----------------------------------

    prompt = f"""
You are an AI assistant for an engineering college.

Use only the following college information to answer the student's question.

College Information:
{college_info}

Student Question:
{question}

Give only a short and clear answer.
Do not repeat the question.
Do not repeat the college information.
"""


    # -----------------------------------
    # Tokenization
    # -----------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )


    # -----------------------------------
    # Generate AI Answer
    # -----------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )


    # -----------------------------------
    # Convert Output to Text
    # -----------------------------------

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    return answer


# -----------------------------------
# Gradio User Interface
# -----------------------------------

interface = gr.Interface(

    fn=chatbot,

    inputs=gr.Textbox(
        label="Student Question",
        placeholder="Ask your college-related question..."
    ),

    outputs=gr.Textbox(
        label="AI Chatbot Answer"
    ),

    title="Engineering College AI Chatbot",

    description=(
        "Ask questions about courses, admission, examinations, "
        "library, hostel, transport and other college facilities."
    )
)


# -----------------------------------
# Start Application
# -----------------------------------

if __name__ == "__main__":

    interface.launch()