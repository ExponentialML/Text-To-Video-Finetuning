import argparse
import os
import warnings
from pathlib import Path
from uuid import uuid4
from utils.lora import inject_inferable_lora
import torch
from diffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline
from models.unet_3d_condition import UNet3DConditionModel
from einops import rearrange
from torch.nn.functional import interpolate

from train import export_to_video, handle_memory_attention, load_primary_models
from utils.lama import inpaint_watermark


def initialize_pipeline(model, device="cuda", xformers=False, sdp=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, _unet = load_primary_models(model)
        del _unet #This is a no op
        unet = UNet3DConditionModel.from_pretrained(model, subfolder='unet')
        unet.disable_gradient_checkpointing()
        
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


def vid2vid(
    pipeline, init_video, init_weight, prompt, negative_prompt, height, width, num_inference_steps, guidance_scale
):
    num_frames = init_video.shape[2]
    init_video = rearrange(init_video, "b c f h w -> (b f) c h w")
    latents = pipeline.vae.encode(init_video).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=num_frames)
    latents = pipeline.scheduler.add_noise(
        original_samples=latents * 0.18215,
        noise=torch.randn_like(latents),
        timesteps=(torch.ones(latents.shape[0]) * pipeline.scheduler.num_train_timesteps * (1 - init_weight)).long(),
    )
    if latents.shape[0] != len(prompt):
        latents = latents.repeat(len(prompt), 1, 1, 1, 1)

    do_classifier_free_guidance = guidance_scale > 1.0

    prompt_embeds = pipeline._encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=latents.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    pipeline.scheduler.set_timesteps(num_inference_steps, device=latents.device)
    timesteps = pipeline.scheduler.timesteps
    timesteps = timesteps[round(init_weight * len(timesteps)) :]

    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        for t in timesteps:
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # reshape latents
            bsz, channel, frames, width, height = latents.shape
            latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            # reshape latents back
            latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

            progress_bar.update()

    video_tensor = pipeline.decode_latents(latents)

    return video_tensor


@torch.inference_mode()
def inference(
    model,
    prompt,
    negative_prompt=None,
    batch_size=1,
    num_frames=16,
    width=256,
    height=256,
    num_steps=50,
    guidance_scale=9,
    init_video=None,
    init_weight=0.5,
    device="cuda",
    xformers=False,
    sdp=False,
    lora_path='',
    lora_rank=64
):
    with torch.autocast(device, dtype=torch.half):
        pipeline = initialize_pipeline(model, device, xformers, sdp)
        inject_inferable_lora(pipeline, lora_path, r=lora_rank)
        prompt = [prompt] * batch_size
        negative_prompt = ([negative_prompt] * batch_size) if negative_prompt is not None else None

        if init_video is not None:
            videos = vid2vid(
                pipeline=pipeline,
                init_video=init_video.to(device=device, dtype=torch.half),
                init_weight=init_weight,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )

        else:
            videos = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                output_type="pt",
            ).frames

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
    parser.add_argument("-T", "--num-frames", type=int, default=16)
    parser.add_argument("-W", "--width", type=int, default=256)
    parser.add_argument("-H", "--height", type=int, default=256)
    parser.add_argument("-s", "--num-steps", type=int, default=50)
    parser.add_argument("-g", "--guidance-scale", type=float, default=9)
    parser.add_argument("-i", "--init-video", type=str, default=None)
    parser.add_argument("-iw", "--init-weight", type=float, default=0.5)
    parser.add_argument("-f", "--fps", type=int, default=8)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-x", "--xformers", action="store_true")
    parser.add_argument("-S", "--sdp", action="store_true")
    parser.add_argument("-lP", "--lora_path", type=str, default="")
    parser.add_argument("-lR", "--lora_rank", type=int, default=64)
    parser.add_argument("-rw", "--remove-watermark", action="store_true")
    args = vars(parser.parse_args())

    output_dir = args.pop("output_dir")
    prompt = args.get("prompt")
    fps = args.pop("fps")
    remove_watermark = args.pop("remove_watermark")
    init_video = args.pop("init_video")

    if init_video is not None:
        vr = decord.VideoReader(init_video)
        init = rearrange(vr[:], "f h w c -> c f h w").div(127.5).sub(1).unsqueeze(0)
        init = interpolate(init, size=(args["num_frames"], args["height"], args["width"]), mode="trilinear")
        args["init_video"] = init

    videos = inference(**args)

    os.makedirs(output_dir, exist_ok=True)
    out_stem = f"{output_dir}/"
    if init_video is not None:
        out_stem += f"({Path(init_video).stem}) * {args['init_weight']} | "
    out_stem += f"{prompt}"

    for video in videos:

        if remove_watermark:
            video = rearrange(video, "c f h w -> f c h w").add(1).div(2)
            video = inpaint_watermark(video)
            video = rearrange(video, "f c h w -> f h w c").clamp(0, 1).mul(255)

        else:
            video = rearrange(video, "c f h w -> f h w c").clamp(-1, 1).add(1).mul(127.5)

        video = video.byte().cpu().numpy()

        export_to_video(video, f"{out_stem} {str(uuid4())[:8]}.mp4", fps)
