import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import subprocess
import tempfile

from typing import Dict, Optional, Tuple, List

import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers

from pkg_resources import resource_filename
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from utils.dataset import VideoDataset
from einops import rearrange, repeat

already_printed_unet = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def create_output_folders(output_dir, config):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"train_{now}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir


def load_primary_models(pretrained_model_path):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    unet = UNet3DConditionModel()

    model_path = os.path.join(os.getcwd(), pretrained_model_path, 'unet', 'diffusion_pytorch_model.bin')
    # Load the pretrained weights
    pretrained_dict = torch.load(
        model_path,
        map_location=torch.device('cuda'),
    )
    unet.load_state_dict(pretrained_dict, strict=False)

    unet.infinet._init_weights()

    unet.infinet.diffusion_depth = 1
    #unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def main():
    pretrained_model_path = "models/model_scope_diffusers"
    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path)

    vae.to("cuda")
    unet.to("cuda")
    text_encoder.to("cuda")

    # Enable VAE slicing to save memory.
    vae.enable_slicing()


    #unet.eval()
    #text_encoder.eval()

    pipeline = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet
    )

    pipeline.enable_xformers_memory_efficient_attention()

    diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = diffusion_scheduler

    prompt = "Couple walking on the beach"
    os.makedirs("samples", exist_ok=True)
    out_file = f"samples/eval_{prompt}.mp4"

    with torch.no_grad():
        video_frames = pipeline(
            prompt,
            width=512,
            height=384,
            num_frames=20,
            num_inference_steps=50,
            guidance_scale=7.5
        ).frames
    video_path = export_to_video(video_frames, out_file)

    del pipeline
    gc.collect()

from PIL import Image
import cv2
def export_to_video(video_frames: List[np.ndarray], output_video_path: str = None, fps: int = 8) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    #fps = 8
    h, w, c = video_frames[0].shape

    os.makedirs(os.path.join(os.getcwd(), 'out'), exist_ok=True)
    for i in range(len(video_frames)):
#        Image.fromarray(video_frames[i]).save(os.path.join(os.getcwd(), 'out', f"frame_{i}.png"))
        cv2.imwrite(os.path.join(os.getcwd(), 'out',
                    f"{i:06}.png"), video_frames[i])

    # create a pipe for ffmpeg to write the video frames to
    ffmpeg_pipe = subprocess.Popen(
        [
            "ffmpeg",
            "-y",  # overwrite output file if it already exists
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-vcodec", "libx264",
            "-preset", "medium",
            "-crf", "23",
            output_video_path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # write each video frame to the ffmpeg pipe
    for frame in video_frames:
        ffmpeg_pipe.stdin.write(frame.tobytes())

    # close the ffmpeg pipe and wait for it to finish writing the video file
    ffmpeg_pipe.stdin.close()
    ffmpeg_pipe.wait()

    return output_video_path

if __name__ == "__main__":
    main()

def find_ffmpeg_binary():
    try:
        import google.colab
        return 'ffmpeg'
    except:
        pass
    for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
        try:
            package_path = resource_filename(package, 'binaries')
            files = [os.path.join(package_path, f) for f in os.listdir(
                package_path) if f.startswith("ffmpeg-")]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0] if files else 'ffmpeg'
        except:
            return 'ffmpeg'