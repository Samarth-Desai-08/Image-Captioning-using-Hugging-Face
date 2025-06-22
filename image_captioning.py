# image_captioning.py

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Load model, processor, and tokenizer
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate caption
def generate_caption(image_path):
    # Load and preprocess image
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate caption
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption, image

# Example usage
if __name__ == "__main__":
    img_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"  # Or local path
    caption, img = generate_caption(img_url)
    print("Caption:", caption)
    
    # Optional display
    plt.imshow(img)
    plt.title(caption)
    plt.axis("off")
    plt.show()
