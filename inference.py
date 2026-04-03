import torch
from PIL import Image
import torchvision.transforms as transforms

from src.models.crnn import CRNN
from src.utils.decoder import Decoder


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load model
model = CRNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.eval()


# Decoder
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
decoder = Decoder(chars)


# Transform (same as training!)
transform = transforms.Compose([
    transforms.Resize((32, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict(image_path):
    # Load image
    image = Image.open(image_path).convert("L")
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0).to(device)  # [1, 1, 32, 160]

    # Forward
    with torch.no_grad():
        outputs = model(image)  # [T, 1, C]

    # Decode
    pred_text = decoder.decode(outputs)

    return pred_text[0]


if __name__ == "__main__":
    img_path = "testIMG.jpg"  #  replace with your image
    text = predict(img_path)
    print("Prediction:", text)