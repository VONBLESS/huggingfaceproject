import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16  # or float16 if bfloat16 gives issues
).to("cuda")
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = (
    "Ultra-realistic full-body portrait of an adult woman standing barefoot in a golden meadow at sunset, "
    "but posed tastefully with soft silk draped around her hips and over one shoulder, "
    "long auburn hair with loose waves catching the warm light, emerald eyes, porcelain skin with natural texture, "
    "gentle breeze moving her hair and the tall grass, golden-hour rim light, shallow depth of field, "
    "85mm lens look, f/1.8, high dynamic range, subtle film grain, serene expression, "
    "no nudity in explicit pose, elegant fine‑art aesthetic"
)

image = pipe(
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(0)
).images[0]

image.save("flux_output.png")
