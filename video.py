# from diffusers import DiffusionPipeline
# import torch
#
# pipe = DiffusionPipeline.from_pretrained(
#     "damo-vilab/text-to-video-ms-1.7b",
#     torch_dtype=torch.float16,
#     variant="fp16"
# ).to("cuda")
#
# prompt = "A futuristic city with flying cars during sunset"
#
# video_frames = pipe(prompt, num_inference_steps=25).frames
#
# # Save as video
# from diffusers.utils import export_to_video
# export_to_video(video_frames, "generated_video.mp4")


import inspect
import torch
from diffusers.utils import export_to_video

try:
    from diffusers import LTXPipeline
except ImportError as exc:
    raise ImportError(
        "LTXPipeline is not available in your diffusers version. "
        "Upgrade diffusers (and dependencies) and retry."
    ) from exc

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16
)
# Enable CPU offloading to reduce VRAM pressure.
# When offloading, do not manually move the whole pipeline to CUDA.
if hasattr(pipe, "enable_model_cpu_offload"):
    pipe.enable_model_cpu_offload()
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()

prompt = (
    "Photorealistic full-body portrait of a woman, head-to-toe in frame, "
    "standing in a sunlit meadow at golden hour, balanced exposure, clean highlights, "
    "sharp facial features, detailed eyes with catchlights, natural skin texture, "
    "realistic hair strands, 85mm lens look, shallow depth of field, crisp detail, "
    "serene expression, camera at mid distance"
)
negative_prompt = (
    "blurry, soft focus, low detail, smeared face, distorted eyes, warped anatomy, "
    "overexposed highlights, blown highlights, haze, fog, veil artifacts, blurry eyes"
    "plastic skin, waxy skin, painterly, cartoon, jitter, flicker, long neck"
)

# Only pass kwargs supported by the pipeline to avoid runtime errors.
call_sig = inspect.signature(pipe.__call__)
supported = set(call_sig.parameters.keys())
kwargs = {
    "height": 512,
    "width": 512,
    "num_frames": 200,
    "num_inference_steps": 100,
    "guidance_scale": 6.5,
    "negative_prompt": negative_prompt,
}
filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported}

result = pipe(prompt, **filtered_kwargs)
frames = result.frames[0]

video_path = export_to_video(frames, fps=16, output_video_path="anime_rooftop.mp4")
print("Saved to:", video_path)
