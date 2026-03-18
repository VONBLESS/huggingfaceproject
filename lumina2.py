import torch
from diffusers import Lumina2Pipeline

pipe = Lumina2Pipeline.from_pretrained(
     "Alpha-VLLM/Lumina-Image-2.0",
    torch_dtype=torch.bfloat16
).to("cuda")  # Use full GPU

prompt = (
    "A stunning woman, whole body to be displayed and porcelain skin"
)

pipe.safety_checker = None

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=6.0,
    num_inference_steps=50,
    cfg_trunc_ratio=0.1,
    cfg_normalization=True,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]

image.save("lumina_demo.png")
