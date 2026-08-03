import torch
from PIL import Image
import torch.nn.functional as F
from src.data_preprocessing import get_data_transforms
from src.model import load_model

def preprocess_for_inference(image, img_size=224):
    """Preprocesses a PIL image for inference."""
    transforms_dict = get_data_transforms()
    val_transform = transforms_dict['val']
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
    return val_transform(image).unsqueeze(0)

def predict_image(model, image_path, transform=None, device='cpu'):
    """Predicts a single image."""
    model.eval()
    model.to(device)
    
    if transform is None:
        tensor = preprocess_for_inference(image_path)
    else:
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
    tensor = tensor.to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        
    return preds.item(), conf.item()

def predict_batch(model, image_paths, transform=None, device='cpu'):
    """Predicts a batch of images."""
    results = []
    for path in image_paths:
        pred, conf = predict_image(model, path, transform, device)
        results.append({'path': path, 'prediction': pred, 'confidence': conf})
    return results
