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
from glob import glob

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
            use_random_start_idx: bool = False,
            preprocessed: bool = False,
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
        self.sample_iters = 0
        self.original_start_idx = sample_start_idx
        self.use_random_start_idx = use_random_start_idx

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.sample_frame_rate_init = sample_frame_rate

    def load_from_json(self, path):
        # Don't load a JSON file if we're doing single video training
        if os.path.exists(self.single_video_path): return

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
    
    def get_frame_range(self, idx, vr):
        return list(range(idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
    
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
        
    def get_vid_idx(self, vr, vid_data=None):

        if self.use_random_start_idx and self.n_sample_frames == 1:
            
            # Randomize the frame rate at different speeds
            self.sample_frame_rate = random.randint(1, self.sample_frame_rate_init)

            # Randomize start frame so that we can train over multiple parts of the video
            random.seed()
            max_sample_rate = abs((self.n_sample_frames - self.sample_frame_rate) + 2)
            max_frame = abs(len(vr) - max_sample_rate)
            idx = random.randint(1, max_frame)
            
        else:
            if vid_data is not None:
                idx = vid_data['frame_index']
            else:
                idx = 1

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
            sample_index = self.get_sample_idx(idx, vr)

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

            idx = self.get_vid_idx(vr, vid_data)

            # Check if idx is greater than the length of the video.
            if idx >= len(vr):
                idx = 1
                
            # Resolve sample index
            sample_index = self.get_sample_idx(idx, vr)
            
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
        **kwargs
    ):
        self.tokenizer = tokenizer

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

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
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

        return {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": prompt_ids[0], "text_prompt": prompt}
