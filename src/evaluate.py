import os
import argparse
import logging
import torch
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.data_preprocessing import get_data_loaders
from src.model import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(data_dir, model_path, model_name):
    """Evaluates the model and logs metrics to MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_data_loaders(data_dir, batch_size=32)
    
    model = load_model(model_path, model_name, device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    os.makedirs("artifacts", exist_ok=True)
    
    with mlflow.start_run():
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        report = classification_report(all_labels, all_preds, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", report['accuracy'])
        mlflow.log_metric("test_f1_macro", report['macro avg']['f1-score'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        logger.info(f"Test Accuracy: {report['accuracy']:.4f}")
        logger.info("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_path", type=str, default="models/cnn_latest.pt")
    parser.add_argument("--model_name", type=str, default="cnn")
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_path, args.model_name)
