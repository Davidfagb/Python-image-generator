# Python-image-generator
Python Script for Text-to-Image Generation This script uses the diffusers library to download and run a pre-trained text-to-image model called Stable Diffusion. You provide a text description (a "prompt"), and the model generates a corresponding image.

# First, you need to install the required libraries:

 ```py
pip install diffusers transformers torch accelerate
```
Now, here is the Python script:

```py

import torch
from diffusers import StableDiffusionPipeline

# If you have a GPU, the model will run on it automatically. Otherwise, it will use the CPU, which will be much slower.

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load the pre-trained Stable Diffusion model.

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)

# Your text description goes here. Be creative, example can be seen below for reference:

prompt = "a blue dog flying in space, holding a red balloon, photorealistic"

print(f"Generating image for prompt: '{prompt}'")

# Generate the image, the 'guidance_scale' parameter controls how much the output should follow the prompt.
image = pipe(prompt, guidance_scale=8.5).images[0]

# The next line will allow you to save the generated image to a file.

output_filename = "generated_image.png"
image.save(output_filename)

print(f"Image saved as {output_filename}")
```
Now, you have to save the code as a Python file (e.g., generate_image.py).

Open your terminal or command prompt, navigate to the directory where you saved the file, and run the script using the command below:

```py
python generate_image.py
```
Goodluck 
