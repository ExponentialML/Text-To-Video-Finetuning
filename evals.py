import torch
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import inception_v3 #, InceptionV3_Weights
from scipy import linalg

def load_and_preprocess_image(image_path, metric="CLIP"):
    if metric == "FID":
        temp_shape = (299, 299)  # InceptionV3 expects 299x299 inputs
        norm_params = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    else:
        temp_shape = (224, 224)  # Resize to 224x224 for CLIP
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

def main():

    # Define paths to your sets of images
    target_image_paths = ['sample_images/charlie_1.jpeg', 
                          'sample_images/charlie_2.jpeg']
    reference_image_paths = ['sample_images/charlie_1.jpeg', 
                             'sample_images/charlie_2.jpeg']
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the InceptionV3 model with pretrained weights
    inception_model = inception_v3(pretrained=True).to(device)
    # inception_model = inception_v3(weights=InceptionV3_Weights.IMAGENET1K_V1).to(device)

    clip_score = calculate_clip_score(target_image_paths[0], model, preprocess)
    print(f"CLIP Score: {clip_score}")

    fid_score = calculate_fid_score(target_image_paths, reference_image_paths, inception_model, device)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    main()