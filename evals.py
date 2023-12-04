import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import clip
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3

# load an image and preprocess it
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return preprocess(image).unsqueeze(0)

# initialize and load the CLIP model
def initialize_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model architecture
    model = clip.model.CLIP(visual="ViT-B/32", text=clip.model.transformer(), context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12)
    
    # Load the pre-trained weights
    model_path = 'path_to_clip_weights.pt'  # Replace with the path to the downloaded CLIP weights
    model.load_state_dict(torch.load(model_path))

    model = model.to(device).eval()
    return model

# calculate CLIP score
def calculate_clip_score(image, model, text="a photo"):
    image = load_and_preprocess_image(image)

    text = clip.tokenize([text]).to(model.visual.device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs

# calculate FID score
def calculate_fid_score(image, model):
    # TODO: Implement FID calculation using InceptionV3 model and CLIP model.
    # FID calculation involves computing feature vectors using InceptionV3 model
    # and then calculating the Frechet distance between these features and features
    # of a set of real images.
    pass

def main(image_path):
    # Initialize CLIP Model
    model = initialize_clip_model()

    # CLIP Evaluation
    clip_score = calculate_clip_score(image_path, model)
    print(f"CLIP Score: {clip_score}")

    # TODO: FID Evaluation
    # fid_score = calculate_fid_score(image_path, model)
    # print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    main(image_path)
