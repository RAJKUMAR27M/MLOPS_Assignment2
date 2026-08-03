import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
import torch
import uvicorn

from app.utils import load_model_for_serving, preprocess_image, postprocess_prediction

# Configure structured JSON logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and metrics
ml_models = {}
APP_READY = False
STARTUP_ATTEMPTED = False
REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total API requests",
    ["method", "path", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "API request latency in seconds",
    ["path"]
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions by class",
    ["label"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    model_path = os.getenv("MODEL_PATH", "models/cnn_latest.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info({"message": "Starting up", "model_path": model_path, "device": device})
    
    global APP_READY, STARTUP_ATTEMPTED
    STARTUP_ATTEMPTED = True
    APP_READY = False

    try:
        if os.path.exists(model_path):
            model = load_model_for_serving(model_path, device=device)
            ml_models["model"] = model
            ml_models["device"] = device
            APP_READY = True
            logger.info({"message": "Model loaded successfully"})
        else:
            logger.warning({"message": "Model path not found. Startup will remain unhealthy."})
            ml_models["model"] = None
            APP_READY = False
    except Exception as e:
        logger.error({"message": "Failed to load model", "error": str(e)})
        ml_models["model"] = None
        APP_READY = False
        
    yield
    # Shutdown logic
    logger.info({"message": "Shutting down"})
    ml_models.clear()

app = FastAPI(
    title="Cats vs Dogs Image Classification API",
    description="Binary image classification API using PyTorch and FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(path=request.url.path).observe(process_time)

    logger.info({
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(process_time * 1000, 2)
    })

    return response

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Cats vs Dogs Classification API",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    model_loaded = ml_models.get("model") is not None and APP_READY
    if not STARTUP_ATTEMPTED:
        model_loaded = True
    status_code = 200 if model_loaded else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "timestamp": time.time()
        }
    )

@app.get("/metrics")
def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    model = ml_models.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")
    
    device = ml_models.get("device", "cpu")
    
    try:
        contents = await file.read()
        tensor = preprocess_image(contents)
        tensor = tensor.to(device)
        
        with torch.no_grad():
            output = model(tensor)
            
        predicted_class, confidence, probabilities = postprocess_prediction(output)

        PREDICTION_COUNT.labels(label=predicted_class).inc()
            
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }
        
    except Exception as e:
        logger.error({"message": "Error during prediction", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
