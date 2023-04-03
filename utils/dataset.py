import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch

from itertools import islice
from pathlib import Path
from bucketing import sensible_buckets

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids

def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()

def get_text_prompt(
        text_prompt: str = '', 
        fallback_prompt: str= '',
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    try:
        if use_caption:
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file): 
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)
            
            # Return text prompt if no conditions are met.
            if len(text_prompt) > 1:
                return text_prompt
            else:
                return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt

def path_or_prompt(caption_path, prompt):
    if os.path.exists(self.single_caption_path):
        prompt = read_caption_file(caption_path)
    else:
        return prompt
    
def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices

def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)

    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr)

    return video, vr

# https://github.com/ExponentialML/Video-BLIP2-Preprocessor
class VideoJsonDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            base_width: int = 256,
            base_height: int = 256,
            n_sample_frames: int = 4,
            sample_start_idx: int = 1,
            sample_frame_rate: int = 1,
            json_path: str ="./data",
            json_data = None,
            vid_data_key: str = "video_path",
            use_random_start_idx: bool = False,
            preprocessed: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.use_bucketing = use_bucketing
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.vid_data_key = vid_data_key
        self.train_data = self.load_from_json(json_path, json_data)
        self.use_random_start_idx = use_random_start_idx

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def build_json(self, json_data):
        extended_data = []
        for data in json_data['data']:
            for nested_data in data['data']:
                self.build_json_dict(
                    data, 
                    nested_data, 
                    extended_data
                )
        json_data = extended_data
        return json_data

    def build_json_dict(self, data, nested_data, extended_data):
        clip_path = 'clip_path' if 'clip_path' in nested_data else None

        extended_data.append({
            self.vid_data_key: data[self.vid_data_key],
            'frame_index': nested_data['frame_index'],
            'prompt': nested_data['prompt'],
            'clip_path': nested_data['clip_path']
        })
        
    def load_from_json(self, path, json_data):
        try:
            with open(path) as jpath:
                print(f"Loading JSON from {path}")
                json_data = json.load(jpath)

                return self.build_json(json_data)

        except:
            raise ValueError("Invalid JSON")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_frame_range(self, vr):
        return get_video_frames(
            vr, 
            self.sample_start_idx, 
            self.sample_frame_rate, 
            self.n_sample_frames
        )
    
    def get_vid_idx(self, vr, vid_data=None):
        frames = self.n_sample_frames

        if vid_data is not None:
            idx = vid_data['frame_index']
        else:
            idx = self.sample_start_idx

        return idx

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        frame_range = self.get_frame_range(vr)
        frames = vr.get_batch(frame_range)
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        return video

    def train_data_batch(self, index):

        # If we are training on individual clips.
        if 'clip_path' in self.train_data[index]:

            vid_data = self.train_data[index]

            clip_path = vid_data['clip_path']
            
            # Get video prompt
            prompt = vid_data['prompt']

            video, vr = process_video(
                train_data[self.vid_data_key],
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch, 
            )

            prompt_ids = prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids

         # Assign train data
        train_data = self.train_data[index]

        # Initialize resize
        resize = None

        video, vr = process_video(
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch, 
                train_data[self.vid_data_key]
            )

        # Get video prompt
        prompt = train_data['prompt']
        vr.seek(0)

        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'json'

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        else: 
            return 0

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Use default JSON training
        if self.train_data is not None:
            video, prompt, prompt_ids = self.train_data_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example


class SingleVideoDataset(Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            base_width: int = 256,
            base_height: int = 256,
            n_sample_frames: int = 4,
            frame_skip: int = 1,
            use_random_start_idx: bool = False,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            single_caption_path: str = "",
            use_bucketing: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.use_bucketing = use_bucketing
        
        self.n_sample_frames = n_sample_frames
        self.frame_skip = frame_skip
        self.use_random_start_idx = use_random_start_idx

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt
        self.single_caption_path = single_caption_path

        self.width = width
        self.height = height
        self.curr_video = None
        self.sample_frame_rate_init = frame_skip

    def get_sample_frame(self):
        return self.n_sample_frames

    def get_vid_idx(self, vr, vid_data=None):
        frames = self.get_sample_frame()
        if self.use_random_start_idx:
            
            # Randomize the frame rate at different speeds
            self.frame_skip = random.randint(1, self.frame_skip)

            # Randomize start frame so that we can train over multiple parts of the video
            random.seed()
            max_sample_rate = abs((frames - self.frame_skip) + 2)
            max_frame = abs(len(vr) - max_sample_rate)
            idx = random.randint(1, max_frame)
            
        else:
            if vid_data is not None:
                idx = vid_data['frame_index']
            else:
                idx = 1

        return idx

    def get_frame_range(self, idx, vr):
        frames = self.get_sample_frame()
        return list(range(idx, len(vr), self.frame_skip))[:frames]
    
    def get_sample_idx(self, idx, vr):
        # Get the frame idx range based on the get_vid_idx function
        # We have a fallback here just in case we the frame cannot be read
        sample_idx = self.get_frame_range(idx, vr)
        fallback = self.get_frame_range(1, vr)
        
        # Return the result from the get_vid_idx function. This will error out if it cannot be read.
        try:
            vr.get_batch(sample_idx)
            return sample_idx

        # Return the fallback frame range if it fails
        except:
            return fallback

    def single_video_batch(self):
        train_data = self.single_video_path
        if train_data.endswith(self.vid_types):

            if self.curr_video is None:
               # Get closest aspect ratio bucket.
                if self.use_bucketing:
                    vrm = decord.VideoReader(train_data)
                    h, w, _ = vrm[0].shape

                    width, height = sensible_buckets(self.width, self.height, w, h)
                    self.curr_video = decord.VideoReader(train_data, width=width, height=height)

                    del vrm
                else:
                    self.curr_video  = decord.VideoReader(train_data, width=self.width, height=self.height)

            # Load and sample video frames
            vr = self.curr_video

            idx = self.get_vid_idx(vr)

            # Check if idx is greater than the length of the video.
            if idx >= len(vr):
                idx = 1
                
            # Resolve sample index
            sample_index = self.get_sample_idx(idx, vr)

            # Process video and rearrange
            video = vr.get_batch(sample_index)
            video = rearrange(video, "f h w c -> f c h w")


        prompt = path_or_prompt(self.single_caption_path, self.single_video_prompt)
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        if os.path.exists(self.single_video_path): return 1
        return 0

    def __getitem__(self, index):
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        video, prompt, prompt_ids = self.single_video_batch()

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example
    
class ImageDataset(Dataset):
    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption: bool = False,
        image_dir: str = '',
        single_caption_path: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing

        self.image_dir = self.get_images_list(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_caption_path = single_caption_path

        self.width = width
        self.height = height

    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img = train_data

        img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=1)

        prompt = get_text_prompt(
            file_path=train_data,
            fallback_prompt=self.fallback_prompt,
            ext_types=self.img_types,  
            use_caption=True
        )
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Do image training
        if os.path.exists(self.image_dir[0]):
            img, prompt, prompt_ids = self.image_batch(index)

        example = {
            "pixel_values": (img / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example

class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        # Get closest aspect ratio bucket.
        if self.use_bucketing:
            vrm = decord.VideoReader(self.video_files[index])
            h, w, _ = vrm[0].shape

            width, height = sensible_buckets(self.width, self.height, w, h)
            vr = decord.VideoReader(self.video_files[index], width=width, height=height)

            del vrm
        else:
            vr = decord.VideoReader(self.video_files[index], width=self.width, height=self.height)

        native_fps = vr.get_avg_fps()
        every_nth_frame = round(native_fps / self.fps)

        effective_length = len(vr) // every_nth_frame

        if effective_length < self.n_sample_frames:
            return self.__getitem__(random.randint(0, len(self.video_files) - 1))

        effective_idx = random.randint(0, effective_length - self.n_sample_frames)

        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + self.n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if os.path.exists(self.video_files[index].replace(".mp4", ".txt")):
            with open(self.video_files[index].replace(".mp4", ".txt"), "r") as f:
                prompt = f.read()
        else:
            prompt = self.fallback_prompt

        prompt_ids = self.get_prompt_ids(prompt)

        return {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": prompt_ids[0], "text_prompt": prompt, 'dataset': self.__getname__()}
