# MIT License

# Copyright (c) 2023 Hans Brouwer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import os
import warnings
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline, UNet3DConditionModel
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate

from train import export_to_video, handle_memory_attention, load_primary_models
from utils.lama import inpaint_watermark
from utils.lora import inject_inferable_lora


def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet = load_primary_models(model)
        del _unet  # This is a no op
        unet = UNet3DConditionModel.from_pretrained(model, subfolder="unet")

    pipe: TextToVideoSDPipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        unet=unet.to(device=device, dtype=torch.half),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet.disable_gradient_checkpointing()
    handle_memory_attention(xformers, sdp, unet)
    vae.enable_slicing()

    inject_inferable_lora(pipe, lora_path, r=lora_rank)

    return pipe


def prepare_input_latents(
    pipe: TextToVideoSDPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    init_video: Optional[str],
    vae_batch_size: int,
):

    if init_video is None:
        # initialize with random gaussian noise
        scale = pipe.vae_scale_factor
        shape = (batch_size, pipe.unet.config.in_channels, num_frames, height // scale, width // scale)
        latents = torch.randn(shape)

    else:
        # encode init_video to latents
        latents = encode(pipe, init_video, vae_batch_size)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)

    return latents


def encode(pipe: TextToVideoSDPipeline, pixels: Tensor, batch_size: int = 8):
    print("Encoding video frames...")

    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in range(0, pixels.shape[0], batch_size):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents


def decode(pipe: TextToVideoSDPipeline, latents: Tensor, batch_size: int = 8):
    print("Decoding video frames...")

    nf = latents.shape[2]
    latents = rearrange(latents, "b c f h w -> (b f) c h w")

    pixels = []
    for idx in range(0, latents.shape[0], batch_size):
        latents_batch = latents[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = latents_batch.div(pipe.vae.config.scaling_factor)
        pixels_batch = pipe.vae.decode(latents_batch).sample.cpu()
        pixels.append(pixels_batch)
    pixels = torch.cat(pixels)

    pixels = rearrange(pixels, "(b f) c h w -> b c f h w", f=nf)

    return pixels.float()


@torch.inference_mode()
def diffuse(
    pipe: TextToVideoSDPipeline,
    latents: Tensor,
    init_weight: float,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    window_size: int,
):

    device = pipe.device
    order = pipe.scheduler.config.solver_order if "solver_order" in pipe.scheduler.config else pipe.scheduler.order
    do_classifier_free_guidance = guidance_scale > 1.0
    batch_size, _, num_frames, _, _ = latents.shape
    window_size = min(num_frames, window_size)

    prompt_embeds = pipe._encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # set the scheduler to start at the correct timestep
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    start_step = round(init_weight * len(pipe.scheduler.timesteps))
    timesteps = pipe.scheduler.timesteps[start_step:]
    latents = pipe.scheduler.add_noise(
        original_samples=latents, noise=torch.randn_like(latents), timesteps=timesteps[0]
    )

    # manually track previous outputs for the scheduler as we continually change the section of video being diffused
    prev_latents = [None] * order
    prev_latents[-1] = latents

    with pipe.progress_bar(total=len(timesteps) * num_frames // window_size) as progress:

        for i, t in enumerate(timesteps):

            progress.set_description(f"Diffusing timestep {t}...")

            # rotate latents by a random amount (so each timestep has different chunk borders)
            shift = np.random.randint(0, window_size)
            prev_latents = [None if pl is None else torch.roll(pl, shifts=shift, dims=2) for pl in prev_latents]

            for idx in range(0, num_frames, window_size):  # diffuse each chunk individually

                # update scheduler's previous outputs from our own cache
                pipe.scheduler.model_outputs = [
                    None
                    if prev_latents[(i - 1 - o) % order] is None
                    else prev_latents[(i - 1 - o) % order][:, :, idx : idx + window_size, :, :].to(device)
                    for o in range(order, -1, -1)
                ]
                latents_window = pipe.scheduler.model_outputs[-1]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_window] * 2) if do_classifier_free_guidance else latents_window
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents for scheduler
                pipe.scheduler.model_outputs = [
                    None if mo is None else rearrange(mo, "b c f h w -> (b f) c h w")
                    for mo in pipe.scheduler.model_outputs
                ]
                latents_window = rearrange(latents_window, "b c f h w -> (b f) c h w")
                noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")

                # compute the previous noisy sample x_t -> x_t-1
                latents_window = pipe.scheduler.step(noise_pred, t, latents_window).prev_sample

                # reshape latents back for UNet
                latents_window = rearrange(latents_window, "(b f) c h w -> b c f h w", b=batch_size)

                # write diffused latents to output
                if prev_latents[i % order] is None:
                    prev_latents[i % order] = torch.empty_like(prev_latents[(i - 1) % order])
                prev_latents[i % order][:, :, idx : idx + window_size, :, :] = latents_window.cpu()

                progress.update()

    out_latents = prev_latents[i % order]

    return out_latents


@torch.inference_mode()
def inference(
    model: str,
    prompt: List[str],
    negative_prompt: Optional[List[str]] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    window_size: Optional[int] = None,
    vae_batch_size: int = 8,
    num_steps: int = 50,
    guidance_scale: float = 15,
    init_video: Optional[str] = None,
    init_weight: float = 0.5,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    lora_path: str = "",
    lora_rank: int = 64,
):

    with torch.autocast(device, dtype=torch.half):

        # prepare models
        pipe = initialize_pipeline(model, device, xformers, sdp, lora_path, lora_rank)

        # prepare input latents
        init_latents = prepare_input_latents(
            pipe=pipe,
            batch_size=len(prompt),
            num_frames=num_frames,
            height=height,
            width=width,
            init_video=init_video,
            vae_batch_size=vae_batch_size,
        )
        init_weight = init_weight if init_video is not None else 0  # ignore init_weight as there is no init_video!

        # run diffusion
        latents = diffuse(
            pipe=pipe,
            latents=init_latents,
            init_weight=init_weight,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
        )

        # decode latents to pixel space
        videos = decode(pipe, latents, vae_batch_size)

    return videos


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-n", "--negative-prompt", type=str, default=None)
    parser.add_argument("-o", "--output-dir", type=str, default="./output")
    parser.add_argument("-B", "--batch-size", type=int, default=1)
    parser.add_argument("-W", "--width", type=int, default=256)
    parser.add_argument("-H", "--height", type=int, default=256)
    parser.add_argument("-T", "--num-frames", type=int, default=24)
    parser.add_argument("-WS", "--window-size", type=int, default=None)
    parser.add_argument("-VB", "--vae-batch-size", type=int, default=8)
    parser.add_argument("-s", "--num-steps", type=int, default=50)
    parser.add_argument("-g", "--guidance-scale", type=float, default=15)
    parser.add_argument("-i", "--init-video", type=str, default=None)
    parser.add_argument("-iw", "--init-weight", type=float, default=0.5)
    parser.add_argument("-f", "--fps", type=int, default=8)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-x", "--xformers", action="store_true")
    parser.add_argument("-S", "--sdp", action="store_true")
    parser.add_argument("-lP", "--lora_path", type=str, default="")
    parser.add_argument("-lR", "--lora_rank", type=int, default=64)
    parser.add_argument("-rw", "--remove-watermark", action="store_true")
    args = parser.parse_args()

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    out_name = f"{args.output_dir}/"
    if args.init_video is not None:
        out_name += f"({Path(args.init_video).stem}) * {args.init_weight} | "
    out_name += f"{args.prompt}"

    args.prompt = [args.prompt] * args.batch_size
    if args.negative_prompt is not None:
        args.negative_prompt = [args.negative_prompt] * args.batch_size

    if args.window_size is None:
        args.window_size = args.num_frames
    else:
        assert args.num_frames % args.window_size == 0, "num_frames must be exactly divisible by window_size!"

    if args.init_video is not None:
        vr = decord.VideoReader(args.init_video)
        init = rearrange(vr[:], "f h w c -> c f h w").div(127.5).sub(1).unsqueeze(0)
        init = interpolate(init, size=(args.num_frames, args.height, args.width), mode="trilinear")
        args.init_video = init

    # =========================================
    # ============= sample videos =============
    # =========================================

    videos = inference(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        window_size=args.window_size,
        vae_batch_size=args.vae_batch_size,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        init_video=args.init_video,
        init_weight=args.init_weight,
        device=args.device,
        xformers=args.xformers,
        sdp=args.sdp,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
    )

    # =========================================
    # ========= write outputs to file =========
    # =========================================

    os.makedirs(args.output_dir, exist_ok=True)

    for video in videos:

        if args.remove_watermark:
            print("Inpainting watermarks...")
            video = rearrange(video, "c f h w -> f c h w").add(1).div(2)
            video = inpaint_watermark(video)
            video = rearrange(video, "f c h w -> f h w c").clamp(0, 1).mul(255)

        else:
            video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

        video = video.byte().cpu().numpy()

        export_to_video(video, f"{out_name} {str(uuid4())[:8]}.mp4", args.fps)
