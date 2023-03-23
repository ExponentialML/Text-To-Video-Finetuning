# Text-To-Video-Finetuning
## Finetune ModelScope's Text To Video model using Diffusers 🧨 
***(This is a WIP)***


https://user-images.githubusercontent.com/59846140/227371625-07f0ecb7-fed3-49f4-99a4-e5d5abd05a57.mp4

*a small cute dragon sitting on a tree branch in a city, realistc, render, cyberpunk, steampunk*
*Trained on a bird for 1100 steps with last attention layers unfrozen*


## Getting Started
### Requirements

```bash
git clone https://github.com/ExponentialML/Text-To-Video-Finetuning.git
```

```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers
pip install einops
pip install decord
pip install tqdm
```

### Models
The models were downloaded from here https://huggingface.co/damo-vilab/text-to-video-ms-1.7b/tree/main.

This repository was only tested with **FP16** safetensors. Other files (bin, FP32) should work fine, but if you have any trouble, refer to this.

## Hardware
Minimum RTX 3090. You're free to open a PR for optimization (please do!), but this is heavy without gradient checkpointing support.

## Usage
```python
python train.py --config ./configs/my_config.yaml
```

## Preprocessing your data
All videos were preprocessed using the script [here](https://github.com/ExponentialML/Video-BLIP2-Preprocessor) using automatic BLIP2 captions. Please follow the instructions there.

If you wish to use a custom dataloader (for instance, a folder of mp4's and captions), you're free to update the dataloader [here](https://github.com/ExponentialML/Text-To-Video-Finetuning/blob/d72e34cfbd91d2a62c07172f9ef079ca5cd651b2/utils/dataset.py#L83). 

Feel free to share your dataloaders for others to use! It would be much appreciated.

## Configuration
The configuration uses a YAML config borrowed from [Tune-A-Video](https://github.com/showlab/Tune-A-Video) reposotories. Here's the gist of how it works.

<details>
  
```yaml

# The path to your diffusers folder. The structure should look exactly like the huggingface one with folders and json configs
pretrained_model_path: "diffusers_path"

# The directory where your training runs (and samples) will be saved.
output_dir: "./outputs"

# Enable training the text encoder or not.
train_text_encoder: False

# The basis of where your training data is store.
train_data:
  
  # The path to your JSON file using the steps above.
  json_path: "json/train.json"
  
  # Leave this as true for now. Custom configurations are currently not supported.
  preprocessed: True
  
  # Number of frames to sample from the videos. The higher this number, the more VRAM is required (usage is similar to batchsize)
  n_sample_frames: 4
  
  # Choose whether or not to ignore the frame data from the preprocessing step, and shuffle them.
  shuffle_frames: False
  
  # The height and width of training data.
  width: 256      
  height: 256
  
  # At what frame to start the video sampling. Ignores preprocessing frames.
  sample_start_idx: 0
  
  # The rate of sampling frames. This effectively "skips" frames making it appear faster or slower.
  sample_frame_rate: 1
  
  # The key of the video data name. This is to align with any preprocess script changes.
  vid_data_key: "video_path"

  # The video path and prompt for that video for single video training.
  # If enabled, JSON path is ignored
  single_video_path: ""
  single_video_prompt: ""

# This is the data for validation during training. Prompt will override training data prompts.
  sample_preview: True
  prompt: ""
  num_frames: 16
  width: 256
  height: 256
  num_inference_steps: 50
  guidance_scale: 9

# Training parameters
learning_rate: 5e-6
adam_weight_decay: 0
train_batch_size: 1
max_train_steps: 50000

# Allow checkpointing during training (save once every X amount of steps)
checkpointing_steps: 10000

# How many steps during training before we create a sample
validation_steps: 100

# The parameters to unfreeze. As it is now, all attention layers are unfrozen. 
# Unfreezing resnet layers would lead to better quality, but consumes a very large amount of VRAM.
trainable_modules:
  - "attentions"

# Seed for sampling validation
seed: 64

# Use mixed precision for better memory allocation
mixed_precision: "fp16"

# This seems to be incompatible at the moment in my testing.
use_8bit_adam: False

# Currently has no effect.
enable_xformers_memory_efficient_attention: True

```
  </details>

## Running
After training, you can easily run your model by doing the following.

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

my_trained_model_path = "./trained_model_path/"
pipe = DiffusionPipeline.from_pretrained(my_trained_model_path, torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Your prompt based on train data"
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames)
```
