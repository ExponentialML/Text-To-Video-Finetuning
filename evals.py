import torch
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import inception_v3 
from scipy import linalg
from moviepy.editor import VideoFileClip
import random
import os

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

def calculate_clip_score(image_path, model, preprocess, text="a photo"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = load_and_preprocess_image(image_path)
    image = image.to(device)
    
    # Debug: Print the shape of the image tensor
    print("Shape of preprocessed image:", image.shape)

    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs

def calculate_fid_score(image_paths1, image_paths2, inception_model, device):
    # Process and extract features for the first set of images
    features1 = []
    for image_path in image_paths1:
        image = load_and_preprocess_image(image_path, "FID").to(device)
        features1.append(inception_features(image, inception_model))
    features1 = np.concatenate(features1, axis=0)

    # Process and extract features for the second set of images
    features2 = []
    for image_path in image_paths2:
        image = load_and_preprocess_image(image_path, "FID").to(device)
        features2.append(inception_features(image, inception_model))
    features2 = np.concatenate(features2, axis=0)

    # Calculate mean and covariance for both sets of features
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

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

def extract_frames_every_half_second(video_path, output_dir, sample_rate=2):
    # sample rate is the number of frames taken per second
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clip = VideoFileClip(video_path)
    duration = clip.duration

    # generate times at intervals
    times = [i * (1/sample_rate) for i in range(int(duration / 0.5))]

    for i, time in enumerate(times):
        # extract the frame at the specified time
        frame = clip.get_frame(time)
        output_file_path = os.path.join(output_dir, f'frame_{i}.jpeg')
        clip.save_frame(output_file_path, t=time)
        # print(f"Frame {i} (at time {time}s) written to {output_file_path}")

def get_filenames(directory):
    try:
        files_and_dirs = os.listdir(directory)
        filenames = [directory + f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
        return filenames
    except FileNotFoundError:
        print(f"The directory {directory} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def main():

    target_video_path = 'input/v_SoccerJuggling_g16_c01.mp4'
    reference_video_path = 'input/v_SoccerJuggling_g16_c01.mp4'
    
    target_output_dir = 'output/target/'
    # extract_random_frames(target_video_path, target_output_dir)
    extract_frames_every_half_second(target_video_path, target_output_dir)
    
    reference_output_dir = 'output/reference/'
    # extract_random_frames(reference_video_path, reference_output_dir)
    extract_frames_every_half_second(reference_video_path, reference_output_dir)
    
    target_image_paths = get_filenames(target_output_dir)
    reference_image_paths = get_filenames(reference_output_dir)  # Assuming the same number of images in each set
    # print("target files: ", target_image_paths)
    # print("reference files: ", reference_image_paths)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    inception_model = inception_v3(pretrained=True).to(device)
    
    clip_score = calculate_clip_score(target_image_paths[0], model, preprocess)
    print(f"CLIP Score: {clip_score}")

    fid_score = calculate_fid_score(target_image_paths, reference_image_paths, inception_model, device)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    main()