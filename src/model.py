import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN for Cats vs Dogs."""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LogisticRegressionModel(nn.Module):
    """Baseline Logistic Regression."""
    def __init__(self, input_dim=224*224*3, num_classes=2):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

def get_model(model_name='cnn', num_classes=2):
    """Factory function for models."""
    if model_name == 'cnn':
        return SimpleCNN(num_classes)
    elif model_name == 'logistic':
        return LogisticRegressionModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_model(model_path, model_name='cnn', device='cpu'):
    """Load a saved model."""
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
