import torch
import soundfile as sf
import gradio as gr

from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)

print("Loading Text-to-Speech AI model...")

processor = SpeechT5Processor.from_pretrained(
    "microsoft/speecht5_tts"
)

model = SpeechT5ForTextToSpeech.from_pretrained(
    "microsoft/speecht5_tts"
)

vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan"
)

torch.manual_seed(42)

speaker_embedding = torch.randn(1, 512)

print("Text-to-Speech AI model loaded successfully!")


def text_to_speech(text):

    if text is None or text.strip() == "":
        return None

    try:

        inputs = processor(
            text=text,
            return_tensors="pt"
        )

        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embedding,
            vocoder=vocoder
        )

        output_file = "engineering_speech.wav"

        sf.write(
            output_file,
            speech.numpy(),
            16000
        )

        return output_file

    except Exception as e:

        print("Error:", e)

        return None


interface = gr.Interface(
    fn=text_to_speech,

    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter engineering-related text...",
        label="Engineering Text"
    ),

    outputs=gr.Audio(
        label="Generated Speech"
    ),

    title="Engineering Text-to-Speech Application",

    description=(
        "Enter engineering-related text and "
        "the pre-trained SpeechT5 AI model "
        "will convert the text into speech."
    )
)


if __name__ == "__main__":
    interface.launch()