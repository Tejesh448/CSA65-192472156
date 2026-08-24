import torch
from diffusers import DiffusionPipeline


print("Loading text-to-image model...")

model_id = "segmind/tiny-sd"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

print("Model loaded successfully!")


# -----------------------------------
# Prompt 1 - Simple Bridge
# -----------------------------------

prompt1 = """
A modern suspension bridge over a river,
clear sky, realistic engineering structure
"""


# -----------------------------------
# Prompt 2 - Detailed Bridge
# -----------------------------------

prompt2 = """
A highly detailed modern suspension bridge over a wide river,
strong steel cables, large concrete pillars,
realistic civil engineering design,
professional engineering visualization
"""


# -----------------------------------
# Prompt 3 - Night Bridge
# -----------------------------------

prompt3 = """
A modern suspension bridge over a river at night,
beautiful city lights, illuminated steel cables,
realistic engineering structure,
high quality engineering visualization
"""


# -----------------------------------
# Generate Image 1
# -----------------------------------

print("Generating Image 1...")

image1 = pipe(
    prompt1,
    num_inference_steps=10
).images[0]

image1.save("bridge_prompt_1.png")

print("Image 1 saved as bridge_prompt_1.png")


# -----------------------------------
# Generate Image 2
# -----------------------------------

print("Generating Image 2...")

image2 = pipe(
    prompt2,
    num_inference_steps=10
).images[0]

image2.save("bridge_prompt_2.png")

print("Image 2 saved as bridge_prompt_2.png")


# -----------------------------------
# Generate Image 3
# -----------------------------------

print("Generating Image 3...")

image3 = pipe(
    prompt3,
    num_inference_steps=10
).images[0]

image3.save("bridge_prompt_3.png")

print("Image 3 saved as bridge_prompt_3.png")


print("-----------------------------------")
print("All three images generated successfully!")
print("-----------------------------------")