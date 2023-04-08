# Text-To-Video-Finetuning
## Finetune ModelScope's Text To Video model using Diffusers 🧨 

### Updates
- **2023-3-29**: Added gradient checkpointing support. 
- **2023-3-27**: Support for using Scaled Dot Product Attention for Torch 2.0 users. 

## Getting Started

### Requirements & Installation

```bash
git clone https://github.com/ExponentialML/Text-To-Video-Finetuning.git
cd Text-To-Video-Finetuning
git lfs install
git clone https://huggingface.co/damo-vilab/text-to-video-ms-1.7b ./models/model_scope_diffusers/
```

### Create Conda Environment (Optional)
It is recommended to install Anaconda.

**Windows Installation:** https://docs.anaconda.com/anaconda/install/windows/

**Linux Installation:** https://docs.anaconda.com/anaconda/install/linux/

```bash
conda create -n text2video-finetune python=3.10
conda activate text2video-finetune
```

### Python Requirements
```bash
pip install -r requirements.txt
```

## Hardware

All code was tested on Python 3.10.9 & Torch version 1.13.1 & 2.0.

It is **highly recommended** to install >= Torch 2.0. This way, you don't have to install Xformers *or* worry about memory performance. 

If you don't have Xformers enabled, you can follow the instructions here: https://github.com/facebookresearch/xformers


Recommended to use a RTX 3090, but you should be able to train on GPUs with <= 16GB ram with:
- Validation turned off.
- Xformers or Torch 2.0 Scaled Dot-Product Attention 
- Gradient checkpointing enabled. 
- Resolution of 256.
- Enable all LoRA options.

## Preprocessing your data

### Using Captions

You can use caption files when training on images or video. Simply place them into a folder like so:

**Images**: `/images/img.png /images/img.txt`
**Videos**: `/videos/vid.mp4 | /videos/vid.txt`

Then in your config, make sure to have `-folder` enabled, along with the root directory containing the files.

### Process Automatically

You can automatically caption the videos using the [Video-BLIP2-Preprocessor Script](https://github.com/ExponentialML/Video-BLIP2-Preprocessor)

## Configuration

The configuration uses a YAML config borrowed from [Tune-A-Video](https://github.com/showlab/Tune-A-Video) reposotories. 

All configuration details are placed in `configs/v2/high_vram_config.yaml`. Each parameter has a definition for what it does.

### How would you recommend I proceed with making a config with my data?

I highly recommend (I did this myself) going to `configs/v2/high_vram_config.yaml`. Then make a copy of it and name it whatever you wish `my_train.yaml`.

Then, follow each line and configure it for your specific use case. 

The instructions should be clear enough to get you up and running with your dataset, but feel free to ask any questions in the discussion board.

### Finetuning on high VRAM systems.
```python
python train.py --config ./configs/v2/high_vram_config.yaml
```

### Finetuning on low VRAM systems.
```python
python train.py --config ./configs/v2/low_vram_config.yaml
```

### Finetuning on single images.
```python
python train.py --config ./configs/v2/image_training.yaml
```
---

## Training Results

With a lot of data, you can expect training results to show at roughly 2500 steps at a constant learning rate of 5e-6. 

When finetuning on a single video, you should see results in half as many steps.

After training, you should see your results in your output directory. 

By default, it should be placed at the script root under `./outputs/train_<date>`

## Developing

Please feel free to open a pull request if you have a feature implementation or suggesstion! I welcome all contributions.

I've tried to make the code fairly modular so you can hack away, see how the code works, and what the implementations do.

## Deprecation
If you want to use the V1 repository, you can use the branch [here](https://github.com/ExponentialML/Text-To-Video-Finetuning/tree/version/first-release).
