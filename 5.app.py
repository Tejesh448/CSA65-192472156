import gradio as gr
from transformers import pipeline


# -----------------------------------
# Load Pre-trained Whisper Model
# -----------------------------------

print("Loading Speech-to-Text AI model...")

speech_model = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en"
)

print("Speech-to-Text AI model loaded successfully!")


# -----------------------------------
# Speech-to-Text Function
# -----------------------------------

def speech_to_text(audio):

    if audio is None:
        return "Please record an engineering-related question."

    try:
        result = speech_model(audio)

        text = result["text"].strip()

        if not text:
            return "No speech was detected. Please speak clearly and try again."

        return text

    except Exception as e:
        return "Error processing the audio: " + str(e)


# -----------------------------------
# Gradio Interface
# -----------------------------------

interface = gr.Interface(

    fn=speech_to_text,

    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Speak Your Engineering Question"
    ),

    outputs=gr.Textbox(
        label="Converted Text"
    ),

    title="Engineering Speech-to-Text Application",

    description=(
        "Speak an engineering-related question using your microphone. "
        "The pre-trained Whisper AI model will convert your speech into text."
    )
)


# -----------------------------------
# Start Application
# -----------------------------------

if __name__ == "__main__":
    interface.launch()