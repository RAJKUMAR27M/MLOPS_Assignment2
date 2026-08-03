import os
import sys
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
resp = client.get('/health')
print('status_code', resp.status_code)
print(resp.text)
