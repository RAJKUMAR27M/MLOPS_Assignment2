import pytest
from fastapi.testclient import TestClient
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np

from app.main import app, ml_models
from app.utils import SimpleCNN, preprocess_image, postprocess_prediction, IMG_SIZE

client = TestClient(app)

def test_model_creation():
    model = SimpleCNN(num_classes=2)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    output = model(dummy_input)
    assert output.shape == (1, 2)

def test_logistic_regression_model():
    # As a mock for LR model testing output shape
    class MockLR(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * IMG_SIZE * IMG_SIZE, 2)
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.linear(x)
            
    model = MockLR()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    output = model(dummy_input)
    assert output.shape == (1, 2)

def test_preprocess_for_inference():
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    
    tensor = preprocess_image(byte_im)
    assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert tensor.dtype == torch.float32

def test_predict_output_format():
    mock_output = torch.tensor([[2.0, -1.0]]) # Should favor class 0 (cat)
    pred_class, conf, probs = postprocess_prediction(mock_output)
    
    assert pred_class == "cat"
    assert conf > 0.5
    assert "cat" in probs
    assert "dog" in probs
    assert probs["cat"] + probs["dog"] == pytest.approx(1.0)

def test_load_model_supports_legacy_checkpoint(monkeypatch, tmp_path):
    model_path = tmp_path / "legacy_checkpoint.pt"
    model_path.write_bytes(b"legacy")

    expected_state_dict = SimpleCNN(num_classes=2).state_dict()

    def fake_torch_load(path, map_location=None, **kwargs):
        assert str(path) == str(model_path)
        if kwargs.get("weights_only") is False:
            return expected_state_dict
        raise Exception("Weights only load failed")

    monkeypatch.setattr("app.utils.torch.load", fake_torch_load)

    from app.utils import load_model_for_serving

    loaded = load_model_for_serving(str(model_path), device="cpu")
    assert isinstance(loaded, SimpleCNN)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_no_model():
    # Ensure model is None
    ml_models["model"] = None
    
    # Create dummy image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    )
    
    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]

def test_predict_endpoint_with_mock_model():
    # Setup mock model
    ml_models["model"] = SimpleCNN(num_classes=2)
    ml_models["model"].eval()
    ml_models["device"] = "cpu"
    
    # Create dummy image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["prediction"] in ["cat", "dog"]
