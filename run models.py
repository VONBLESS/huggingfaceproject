from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

prompt = (
    "A stunning woman with long auburn hair, emerald eyes, and porcelain skin, "
    "in a silk gown standing in a meadow at sunset, ultra-realistic 2K."
)# Specify 2K resolution (2048x1152 or square 2048x2048)
image = pipe(prompt, height=1920, width=1440).images[0]
image.save("output_hd.png")
