import os
import sys
sys.path.insert(0, os.getcwd())
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)
response = client.get('/health')
print(response.status_code)
print(response.text)
