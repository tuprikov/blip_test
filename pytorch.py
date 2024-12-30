"""
A simple PyTorch model for image classification.
"""
import requests
import torch

from PIL import Image
from torchvision import transforms


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def predict(input_image: Image.Image) -> dict:
    """Predict the class of an image."""
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(input_image)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences
