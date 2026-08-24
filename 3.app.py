import torch
from diffusers import DiffusionPipeline


print("Loading text-to-image model...")

model_id = "segmind/tiny-sd"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

print("Text-to-image model loaded successfully!")


# Engineering text prompt
prompt = """
A modern suspension bridge over a wide river,
strong steel cables, large concrete support pillars,
realistic civil engineering structure,
modern bridge design, detailed engineering construction,
clear blue sky, daylight, professional engineering visualization
"""


print("Generating engineering image...")

image = pipe(
    prompt,
    num_inference_steps=10
).images[0]


# Save the generated image
image.save("engineering_bridge.png")

print("Engineering image generated successfully!")

print("Image saved as engineering_bridge.png")