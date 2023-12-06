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

import torch
import numpy as np

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

def extract_frames_every_half_second(video_path, output_dir, sample_rate=10):
    # sample rate is the number of frames taken per second
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clip = VideoFileClip(video_path)
    duration = clip.duration

    # generate times at intervals
    sr = 1/sample_rate
    times = [i * (sr) for i in range(int(duration / sr))]

    for i, time in enumerate(times):
        # extract the frame at the specified time
        frame = clip.get_frame(time)
        output_file_path = os.path.join(output_dir, f'frame_{i}.jpeg')
        clip.save_frame(output_file_path, t=time)
        # print(f"Frame {i} (at time {time}s) written to {output_file_path}")

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
    
    # TODO: add real = ".mp4"
    baseline = "A dog is running_dog_benchmark_1701400404_900 7c814e22.mp4"
    unique_token = "A dog is running_unique_token_dog_only_1701671770_900 1aa49d2e.mp4"
    l1 = "A dog is running_dog_unique_token_class_preservation_loss_1701399359_900 245d7438.mp4"
    l2 = "A dog is running_unique-token-class-preservation-loss-lambda-0_31701406141_900 679852e4.mp4"
    
    target_video_path = "input/" + baseline
    reference_video_path = "input/" + l2
    test_prompt = "a dog is running"
    print("test prompt: ", test_prompt)
    
    target_output_dir = 'output/target/'
    print("target video path: ", target_video_path)
    extract_frames_every_half_second(target_video_path, target_output_dir)
    
    print("reference video path: ", reference_video_path)
    reference_output_dir = 'output/reference/'
    extract_frames_every_half_second(reference_video_path, reference_output_dir)
    
    target_image_paths = get_filenames(target_output_dir)
    reference_image_paths = get_filenames(reference_output_dir)  # Assuming the same number of images in each set

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    inception_model = inception_v3(pretrained=True).to(device)
    
    clip_score = calculate_clip_scores(target_output_dir, model, preprocess, text=test_prompt)
    print(f"CLIP Score: {clip_score}")

    fid_score = calculate_fid_score(target_image_paths, reference_image_paths, inception_model, device)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    main()