from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
    safety_checker=None,  # Disable NSFW filter
    use_safetensors=True
).to("cuda")

negative_prompt = (
    "lowres, blurry, soft focus, extra limbs, deformed, bad anatomy, oversharpened, noisy, "
    "text, logo, watermark, bad face, asymmetric face, deformed eyes, crossed eyes, "
    "duplicated face, blurry eyes, poorly drawn face"
)


prompt = (
    "Ultra-realistic russian full-body fashion portrait of a woman, "
    "standing in a sunlit meadow at golden hour, natural skin texture, soft rim light, "
    "sharp facial features, detailed eyes with catchlights, defined eyelashes, subtle pores, "
    "85mm lens look, shallow depth of field, high dynamic range"
    "crisp detail, serene expression"
)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=512,
    width=512,
    guidance_scale=6.5,
    num_inference_steps=150,
    generator=torch.Generator("cuda").manual_seed(1234)
).images[0]

image.save("sfw_realistic.png")
