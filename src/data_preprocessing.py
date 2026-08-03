import os
import shutil
import logging
import random
import subprocess
import zipfile
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _organize_kaggle_train_images(source_dir: Path, dest_dir: Path):
    """Converts Kaggle train image layout into cats/ and dogs/ folders."""
    cats_dir = dest_dir / "cats"
    dogs_dir = dest_dir / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in source_dir.glob("*.jpg"):
        lower_name = img_path.name.lower()
        if lower_name.startswith("cat"):
            shutil.copy(img_path, cats_dir / img_path.name)
            copied += 1
        elif lower_name.startswith("dog"):
            shutil.copy(img_path, dogs_dir / img_path.name)
            copied += 1

    return copied


def download_dataset():
    """
    Downloads Cats vs Dogs from Kaggle when credentials are available.
    Falls back to synthetic samples for CI/offline environments.
    """
    logger.info("Downloading dataset...")
    raw_dir = Path("data/raw")
    raw_cats = raw_dir / "cats"
    raw_dogs = raw_dir / "dogs"
    raw_cats.mkdir(parents=True, exist_ok=True)
    raw_dogs.mkdir(parents=True, exist_ok=True)

    # If dataset already exists, reuse it.
    if list(raw_cats.glob("*.jpg")) and list(raw_dogs.glob("*.jpg")):
        logger.info("Existing dataset found in data/raw; skipping download.")
        return

    kaggle_ok = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    download_root = Path("data/kaggle_download")
    download_root.mkdir(parents=True, exist_ok=True)

    if kaggle_ok:
        try:
            cmd = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "dogs-vs-cats",
                "-p",
                str(download_root),
            ]
            logger.info("Attempting Kaggle download for dogs-vs-cats...")
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            competition_zip = download_root / "dogs-vs-cats.zip"
            if competition_zip.exists():
                with zipfile.ZipFile(competition_zip, "r") as zf:
                    zf.extractall(download_root)

            train_zip = download_root / "train.zip"
            if train_zip.exists():
                train_extract_dir = download_root / "train"
                with zipfile.ZipFile(train_zip, "r") as zf:
                    zf.extractall(train_extract_dir)

                copied = _organize_kaggle_train_images(train_extract_dir, raw_dir)
                if copied > 0:
                    logger.info("Kaggle dataset prepared successfully in data/raw")
                    return
        except Exception as exc:
            logger.warning(f"Kaggle download failed, using fallback synthetic data: {exc}")
    else:
        logger.info("Kaggle credentials not set; using fallback synthetic data.")

    for i in range(10):
        img = Image.new(
            "RGB",
            (224, 224),
            color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
        img.save(raw_cats / f"cat_{i}.jpg")
        img.save(raw_dogs / f"dog_{i}.jpg")
    logger.info("Fallback dataset generated at data/raw")

def preprocess_images(raw_dir: str, output_dir: str, img_size: int = 224):
    """
    Resizes images and filters corrupted files.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    
    for category in ['cats', 'dogs']:
        cat_dir = raw_path / category
        out_cat_dir = out_path / category
        out_cat_dir.mkdir(parents=True, exist_ok=True)
        
        if not cat_dir.exists():
            continue
            
        for img_path in cat_dir.glob('*.jpg'):
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((img_size, img_size))
                    img.save(out_cat_dir / img_path.name)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")

def split_dataset(data_dir: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Splits dataset into train/val/test directories.
    """
    random.seed(42)
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    
    for category in ['cats', 'dogs']:
        cat_dir = data_path / category
        if not cat_dir.exists():
            continue
            
        images = list(cat_dir.glob('*.jpg'))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        for split, imgs in splits.items():
            split_dir = out_path / split / category
            split_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, split_dir / img.name)

def get_data_transforms():
    """Returns torchvision transforms dictionary."""
    return {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

def get_data_loaders(data_dir: str, batch_size: int = 32, img_size: int = 224):
    """Returns DataLoader objects."""
    data_transforms = get_data_transforms()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=2)
                   for x in ['train', 'val', 'test']}
    
    return dataloaders
