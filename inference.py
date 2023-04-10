import os
import argparse
import warnings
from uuid import uuid4

import torch
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline
from einops import rearrange

from train import export_to_video, handle_memory_attention, load_primary_models
from utils.lama import inpaint_watermark


def initialize_pipeline(model, device="cuda", xformers=False, sdp=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)

    pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    unet._set_gradient_checkpointing(value=False)
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()
    return pipeline


@torch.inference_mode()
def inference(
    model,
    prompt,
    batch_size=1,
    num_frames=16,
    width=256,
    height=256,
    num_steps=50,
    guidance_scale=9,
    device="cuda",
    xformers=False,
    sdp=False,
):
    with torch.autocast(device, dtype=torch.half):
        pipeline = initialize_pipeline(model, device, xformers, sdp)

        videos = pipeline(
            prompt=[prompt] * batch_size,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).frames

        return videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, default="./output")
    parser.add_argument("-B", "--batch-size", type=int, default=1)
    parser.add_argument("-T", "--num-frames", type=int, default=16)
    parser.add_argument("-W", "--width", type=int, default=256)
    parser.add_argument("-H", "--height", type=int, default=256)
    parser.add_argument("-s", "--num-steps", type=int, default=50)
    parser.add_argument("-g", "--guidance-scale", type=float, default=9)
    parser.add_argument("-f", "--fps", type=int, default=8)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-x", "--xformers", action="store_true")
    parser.add_argument("-S", "--sdp", action="store_true")
    parser.add_argument("-rw", "--remove-watermark", action="store_true")
    args = vars(parser.parse_args())

    output_dir = args.pop("output_dir")
    prompt = args.get("prompt")
    fps = args.pop("fps")
    remove_watermark = args.pop("remove_watermark")

    videos = inference(**args)

    os.makedirs(output_dir, exist_ok=True)

    for video in videos:

        if remove_watermark:
            video = rearrange(video, "c f h w -> f c h w").add(1).div(2)
            video = inpaint_watermark(video)
            video = rearrange(video, "f c h w -> f h w c").clamp(0, 1).mul(255)

        else:
            video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

        video = video.byte().cpu().numpy()

        export_to_video(video, f"{output_dir}/{prompt} {str(uuid4())[:8]}.mp4", fps)
