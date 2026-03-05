import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

prompt = (
    "Ultra-realistic full-body fashion portrait of a woman in a flowing silk skirt, "
    "standing in a sunlit meadow at golden hour, gentle breeze moving the fabric, "
    "natural skin texture, soft rim light, warm sunlight with subtle lens bloom, "
    "sharp facial features, detailed eyes with catchlights, defined eyelashes, subtle pores, "
    "cinematic 85mm lens look, shallow depth of field, high dynamic range, crisp detail, "
    "serene expression, stable framing, smooth motion"
)
negative_prompt = (
    "lowres, blurry, extra limbs, deformed, bad anatomy, jittery, flicker, noisy, "
    "watermark, text, logo, oversharpened, plastic skin, harsh shadows"
)

generator = torch.Generator(device="cuda").manual_seed(1234)
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=45,
    guidance_scale=6.5,
    height=320,
    width=576,
    num_frames=48,
    generator=generator
)

frames = result.frames[0]
video_path = export_to_video(frames, fps=12, output_video_path="anime_rooftop_v2.mp4")
print("Saved to:", video_path)
