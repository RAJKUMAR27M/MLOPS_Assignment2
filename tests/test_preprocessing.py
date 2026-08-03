import os
import torch
import pytest
from PIL import Image
import numpy as np
from torchvision import transforms

def test_image_resize():
    # Create random PIL image
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Resize
    img_resized = img.resize((224, 224))
    
    # Assert
    assert img_resized.size == (224, 224)

def test_data_transforms_train():
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = train_transforms(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

def test_data_transforms_val():
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = val_transforms(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

def test_split_ratios(tmp_path):
    # Create mock directories
    cats_dir = tmp_path / "cats"
    cats_dir.mkdir()
    dogs_dir = tmp_path / "dogs"
    dogs_dir.mkdir()
    
    # Create dummy files
    for i in range(10):
        (cats_dir / f"cat_{i}.jpg").touch()
        (dogs_dir / f"dog_{i}.jpg").touch()
        
    cats_files = list(cats_dir.glob("*.jpg"))
    dogs_files = list(dogs_dir.glob("*.jpg"))
    
    assert len(cats_files) == 10
    assert len(dogs_files) == 10
    
    # Simple manual split logic check
    train_ratio = 0.8
    train_cats = cats_files[:int(len(cats_files) * train_ratio)]
    val_cats = cats_files[int(len(cats_files) * train_ratio):]
    
    assert len(train_cats) == 8
    assert len(val_cats) == 2

def test_corrupted_image_handling(tmp_path):
    bad_img = tmp_path / "bad.jpg"
    with open(bad_img, "wb") as f:
        f.write(b"not an image")
        
    with pytest.raises(Exception):
        Image.open(bad_img).verify()
