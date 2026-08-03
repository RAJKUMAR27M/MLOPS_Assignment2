import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.data_preprocessing import get_data_loaders
from src.model import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_dir, model_name, epochs, batch_size, lr):
    """Trains the model with MLflow tracking."""
    mlflow.set_experiment("cats_vs_dogs")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    dataloaders = get_data_loaders(data_dir, batch_size=batch_size)
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr
        })
        
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_acc", epoch_acc.item(), step=epoch)
                
                if phase == 'train':
                    train_losses.append(epoch_loss)
                else:
                    val_losses.append(epoch_loss)
                    
                logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                
        # Save model
        model_path = os.path.join("models", f"{model_name}_latest.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        # Plot loss
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss Curves')
        loss_plot_path = "artifacts/loss_curves.png"
        plt.savefig(loss_plot_path)
        mlflow.log_artifact(loss_plot_path)
        plt.close()
        
        logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.model_name, args.epochs, args.batch_size, args.lr)
