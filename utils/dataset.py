import os
import decord
import numpy as np
import random
import json
import torchvision.transforms as T
import torch
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange

class VideoDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            json_path: str ="./data",
            vid_data_key: str = "video_path",
            preprocessed: bool = False,
            shuffle_frames: bool = False,
            use_vision_model: bool = False,
            single_video_path: str = "",
            single_video_prompt: str = "",
            **kwargs
    ):

        self.tokenizer = tokenizer
        self.preprocessed = preprocessed

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt

        self.train_data = self.load_from_json(json_path)
        self.vid_data_key = vid_data_key
        self.shuffle_frames = shuffle_frames
        self.sample_iters = 0
        self.original_start_idx = sample_start_idx

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.sample_frame_rate_init = sample_frame_rate

    def load_from_json(self, path):
        # Don't load a JSON file if we're doing single video training
        if os.path.exists(single_video_path): return

        try:
            print(f"Loading JSON from {path}")
            with open(path) as jpath:
                json_data = json.load(jpath)
            
            if not self.preprocessed:
                for data in json_data['data']:
                    is_valid = self.validate_json(json_data['base_path'],data["folder"])
                    if not is_valid:
                        raise ValueError(f"{data['folder']} is not a valid folder for path {json_data['base_path']}.")

                print(f"{json_data['name']} successfully loaded.")
                return json_data
            else: 
                print("Preprocessed mode.")
                return json_data

        except:
            raise ValueError("Invalid JSON")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_prompt_ids(self, prompt):
        prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
        ).input_ids

        return prompt_ids

    def get_vid_idx(self, vr):
        # Randomize the frame rate at different speeds
        self.sample_frame_rate = random.randint(1, self.sample_frame_rate_init)

        # Randomize start frame so that we can train over multiple parts of the video
        random.seed()
        max_sample_rate = abs((self.n_sample_frames - self.sample_frame_rate) + 2)
        max_frame = abs(len(vr) - max_sample_rate)
        idx = random.randint(1, max_frame)

        return idx

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data['data'])
        else:
            return 1

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Check if we're doing single video training
        if os.path.exists(self.single_video_path):
            train_data = self.single_video_path

            # Load and sample video frames
            vr = decord.VideoReader(train_data, width=self.width, height=self.height)

            idx = self.get_vid_idx(vr)

            # Check if idx is greater than the length of the video.
            if idx >= len(vr):
                idx = 1
                
            # Resolve sample index
            sample_index = list(range(idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]

            # Process video and rearrange
            video = vr.get_batch(sample_index)
            video = rearrange(video, "f h w c -> f c h w")

            prompt = self.single_video_prompt
            prompt_ids = self.get_prompt_ids(prompt)
        
        # Use default JSON training
        else:
            # Assign train data
            train_data = self.train_data['data'][index]

            # load and sample video frames
            vr = decord.VideoReader(train_data[self.vid_data_key], width=self.width, height=self.height)

            # Pick a random video from the dataset
            vid_data = random.choice(train_data['data'])

            # Set a variable framerate between 1 and 30 FPS 
            random.seed()
            self.sample_frame_rate = random.randint(1, self.sample_frame_rate_init)

            idx = self.get_vid_idx(vr)

            # Check if idx is greater than the length of the video.
            if idx >= len(vr):
                idx = 1
                
            # Resolve sample index
            sample_index = list(range(idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
            
            # Get video prompt
            prompt = vid_data['prompt']

            video = vr.get_batch(sample_index)
            video = rearrange(video, "f h w c -> f c h w")

            prompt_ids = self.get_prompt_ids(prompt)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt
        }

        return example
