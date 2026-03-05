import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16  # or torch.float16 if needed
).to("cuda")
pipe.safety_checker = None

prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt,
    height=384,
    width=384,
    guidance_scale=2.0,
    num_inference_steps=20,
    max_sequence_length=256,
    generator=torch.Generator(device="cuda").manual_seed(0)
).images[0]

image.save("flux-schnell-fast.png")
