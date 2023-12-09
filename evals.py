import torch
import clip
import numpy as np
from functools import partial
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import inception_v3 
from scipy import linalg
from datetime import timedelta
from moviepy.editor import VideoFileClip
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score
import glob
import random
import os
import clip
import argparse

def load_and_preprocess_image(image_path, metric="CLIP"):
    if metric == "FID":
        temp_shape = (299, 299)  # 299x299 for InceptionV3
        norm_params = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    else:
        temp_shape = (224, 224)  # 224x224 for CLIP
        norm_params = [(0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)]
    preprocess = transforms.Compose([
        transforms.Resize(temp_shape),  
        transforms.ToTensor(),
        transforms.Normalize(norm_params[0], norm_params[1]),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def inception_features(images, inception_model):
    inception_model.eval()
    with torch.no_grad():
        return inception_model(images).detach().cpu().numpy()

def calculate_clip_scores(directory, model, preprocess, text="a photo"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_tokens = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    scores = []

    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a directory.")

    image_paths = [os.path.join(directory, file) for file in os.listdir(directory)
                   if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    for image_path in image_paths:
        image = load_and_preprocess_image(image_path)
        image = image.to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

            cos_similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
            scores.append(cos_similarity.cpu().numpy())

    average_score = np.mean(scores)
    return average_score

def load_and_preprocess_image_for_fid(image_path):
    image = Image.open(image_path).convert("RGB")
    # resize the image for Inception model (which is used in FID calculation)
    # use Image.Resampling.LANCZOS for high-quality downsampling
    image = image.resize((299, 299), Image.Resampling.LANCZOS)
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # Change shape from HWC to CHW
    image_tensor = image_tensor.type(torch.uint8)
    return image_tensor

def calculate_fid_score(image_paths1, image_paths2, inception_model, device):
    # process and extract features for the first set of images
    features1 = []
    for image_path in image_paths1:
        image = load_and_preprocess_image(image_path, "FID").to(device)
        features1.append(inception_features(image, inception_model))
    features1 = np.concatenate(features1, axis=0)

    # process and extract features for the second set of images
    features2 = []
    for image_path in image_paths2:
        image = load_and_preprocess_image(image_path, "FID").to(device)
        features2.append(inception_features(image, inception_model))
    features2 = np.concatenate(features2, axis=0)

    # calc mean and covariance for both sets of features
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    features1, features2 = None, None  # free memory
    
    # calc FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_fid_score_with_torchmetrics(image_paths1, image_paths2):
    fid = FrechetInceptionDistance(feature=2048)
    # process images and update FID for the first set
    for image_path in image_paths1:
        image = load_and_preprocess_image_for_fid(image_path)
        fid.update(image.unsqueeze(0), real=True)  # Unsqueeze to add batch dimension
    # process images and update FID for the second set
    for image_path in image_paths2:
        image = load_and_preprocess_image_for_fid(image_path)
        fid.update(image.unsqueeze(0), real=False)  # Unsqueeze to add batch dimension
    # compute FID score
    fid_score = fid.compute()
    # convert from tensor to Python float
    return fid_score.item()  

def extract_random_frames(video_path, output_dir, num_frames=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clip = VideoFileClip(video_path)
    duration = clip.duration

    # generate random times (in seconds)
    times = random.sample(range(int(duration)), num_frames)

    for i, time in enumerate(times):
        # extract the frame at the random time
        frame = clip.get_frame(time)
        output_file_path = os.path.join(output_dir, f'frame_{i}.jpeg')
        clip.save_frame(output_file_path, t=time)
        # print(f"Frame {i} (at time {time}s) written to {output_file_path}")

def extract_frames_every_half_second(video_path, output_dir, r=0, sample_rate=10):
    # sample rate is the number of frames taken per second
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clip = VideoFileClip(video_path)
    duration = clip.duration

    # generate times at intervals
    sr = 1 / sample_rate
    times = [i * sr for i in range(int(duration / sr))]

    for i, time in enumerate(times):
        # Format the time into a readable timestamp (hours, minutes, seconds)
        timestamp = str(timedelta(seconds=round(time)))
        
        # extract the frame at the specified time
        frame = clip.get_frame(time)
        # Include the timestamp in the output file name
        output_file_path = os.path.join(output_dir, f'frame_{i}_{timestamp}_{r}.jpeg')
        clip.save_frame(output_file_path, t=time)
        # print(f"Frame {i} (at time {time}s, timestamp {timestamp}) written to {output_file_path}")

def get_filenames(directory, valid_extensions=['.jpg', '.jpeg', '.png']):
    try:
        files_and_dirs = os.listdir(directory)
        filenames = [
            os.path.join(directory, f) for f in files_and_dirs 
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in valid_extensions
        ]
        return filenames
    except FileNotFoundError:
        print(f"The directory {directory} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def main():

    parser = argparse.ArgumentParser(description='Process video paths and text prompt.')
    parser.add_argument('target_video_path', type=str, help='The path to the target video')
    parser.add_argument('reference_video_path', type=str, help='The path to the reference video')
    parser.add_argument('test_prompt', type=str, help='Text prompt for CLIP score calculation')
    args = parser.parse_args()
    target_video_path = args.target_video_path
    reference_video_path = args.reference_video_path
    test_prompt = args.test_prompt

    # retrieve all video files from the target and reference directories
    target_video_paths = glob.glob(os.path.join(target_video_path, '*.mp4')) # Adjust the extension if needed
    reference_video_paths = glob.glob(os.path.join(reference_video_path, '*.mp4')) # Adjust the extension if needed

    i = 0
    for target_video_path in target_video_paths:
        target_output_dir = os.path.join('output/target', os.path.basename(target_video_path).split('.')[0])
        print("target video path: ", target_video_path)
        extract_frames_every_half_second(target_video_path, 'output/target', i)
        i += 1

    i = 0
    for reference_video_path in reference_video_paths:
        reference_output_dir = os.path.join('output/reference', os.path.basename(reference_video_path).split('.')[0])
        print("reference video path: ", reference_video_path)
        extract_frames_every_half_second(reference_video_path, 'output/reference', i)
        i += 1
        
    print("test prompt: ", test_prompt)
    target_output_dir, reference_output_dir = "output/target", "output/reference"
    target_image_paths = get_filenames(target_output_dir)
    reference_image_paths = get_filenames(reference_output_dir)  # Assuming the same number of images in each set

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    inception_model = inception_v3(pretrained=True).to(device)
    
    clip_score = calculate_clip_scores(target_output_dir, model, preprocess, text=test_prompt)
    print(f"CLIP Score (ours): {clip_score}")

    # clip_score_torch = calculate_clip_scores_with_torchmetrics(target_output_dir, model, preprocess, text=test_prompt)
    # print(f"CLIP Score (torch): {clip_score_torch}")

    # fid_score_ours = calculate_fid_score(reference_image_paths, target_image_paths, inception_model, device)
    # print(f"FID Score (ours): {fid_score_ours}")

    fid_score_torch = calculate_fid_score_with_torchmetrics(reference_image_paths, target_image_paths)
    print(f"FID Score (torch): {fid_score_torch}")
    
if __name__ == "__main__":
    main()