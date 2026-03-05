from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-diffusion-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

prompt = (
    "Ultra-realistic full-body fashion portrait of a woman"
    "standing in a sunlit meadow at golden hour, natural skin texture, soft rim light, "
    "sharp facial features, detailed eyes with catchlights, defined eyelashes, subtle pores, "
    "85mm lens look, shallow depth of field, high dynamic range"
    "crisp detail, serene expression"
)
negative_prompt = (
    "lowres, blurry, soft focus, extra limbs, deformed, bad anatomy, oversharpened, noisy, "
    "text, logo, watermark, bad face, asymmetric face, deformed eyes, crossed eyes, "
    "duplicated face, blurry eyes, poorly drawn face"
)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=512,
    guidance_scale=6.5,
    num_inference_steps=150,
    generator=torch.Generator("cuda").manual_seed(1234)
).images[0]
image.save("output.png")
