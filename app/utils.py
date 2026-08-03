import os
import torch
from torchvision import transforms
from PIL import Image
import io

from src.model import SimpleCNN

CLASS_NAMES = ['cat', 'dog']
IMG_SIZE = 224

def load_model_for_serving(model_path, device='cpu'):
    """
    Loads a PyTorch model from the specified path.
    Uses the same SimpleCNN architecture defined in src/model.py so that
    weights produced by src/train.py load correctly.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Initialize model architecture
    model = SimpleCNN(num_classes=len(CLASS_NAMES))

    # Load weights. Accept both a plain state_dict and a checkpoint dict.
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("layer") or k.startswith("conv") or k.startswith("fc") for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    if isinstance(state_dict, dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

def preprocess_image(image_bytes, img_size=IMG_SIZE):
    """
    Converts image bytes to a preprocessed PyTorch tensor.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image)
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    return tensor

def postprocess_prediction(output):
    """
    Converts model raw output to class label and probabilities.
    """
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    
    prob_dict = {
        CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities)
    }
    
    return predicted_class, confidence.item(), prob_dict
