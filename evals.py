import torch
import clip
from PIL import Image
import torchvision.transforms as transforms

def load_and_preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def calculate_clip_score(image_path, model, preprocess, text="a photo"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = load_and_preprocess_image(image_path)
    image = image.to(device)

    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs

def main(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    clip_score = calculate_clip_score(image_path, model, preprocess)
    print(f"CLIP Score: {clip_score}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evals.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)
