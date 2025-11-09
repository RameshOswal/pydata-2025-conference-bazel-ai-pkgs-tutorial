"""
ML Model Serving API using FastAPI and OpenAPI.

This FastAPI application provides REST endpoints for:
1. Model predictions
2. Model metadata and health checks
3. Batch predictions
4. Model management (loading/switching models)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API request/response schemas
class PumpkinFeatures(BaseModel):
    """Input features for pumpkin price prediction."""
    day_of_year: int = Field(..., ge=1, le=366, description="Day of the year (1-366)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    variety: str = Field(..., description="Pumpkin variety", example="PIE TYPE")
    city: str = Field(..., description="City name", example="BOSTON")
    package: str = Field(..., description="Package type", example="bushel cartons")
    
    class Config:
        schema_extra = {
            "example": {
                "day_of_year": 280,
                "month": 10,
                "variety": "PIE TYPE",
                "city": "BOSTON",
                "package": "bushel cartons"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request containing multiple samples."""
    model_config = {"protected_namespaces": ()}
    
    samples: List[PumpkinFeatures] = Field(..., min_items=1, max_items=1000)
    model_name: Optional[str] = Field(None, description="Specific model to use")

class PredictionResponse(BaseModel):
    """Single prediction response."""
    model_config = {"protected_namespaces": ()}
    
    predicted_price: float = Field(..., description="Predicted pumpkin price")
    confidence_interval: Optional[List[float]] = Field(None, description="95% confidence interval")
    model_used: str = Field(..., description="Name of the model used")
    prediction_timestamp: datetime = Field(..., description="When the prediction was made")
    features_used: Dict[str, Any] = Field(..., description="Processed features used for prediction")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    model_config = {"protected_namespaces": ()}
    
    predictions: List[PredictionResponse]
    batch_size: int
    processing_time_ms: float
    model_used: str

class ModelInfo(BaseModel):
    """Model metadata information."""
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    model_type: str
    feature_names: List[str]
    performance_metrics: Dict[str, float]
    training_data_shape: Union[str, List[int]]
    last_updated: datetime
    is_active: bool

class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    models_loaded: int
    active_model: str
    uptime_seconds: float

# Global model storage
class ModelStore:
    """Centralized model storage and management."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.active_model: str = None
        self.start_time = datetime.now()
    
    def load_model(self, model_path: str, model_name: str, metadata: Dict) -> bool:
        """Load a model from disk."""
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            self.metadata[model_name] = metadata
            
            if self.active_model is None:
                self.active_model = model_name
                
            logger.info(f"Loaded model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def get_model(self, model_name: str = None):
        """Get a specific model or the active model."""
        if model_name is None:
            model_name = self.active_model
        
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return self.models[model_name], self.metadata[model_name]
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self.models.keys())
    
    def set_active_model(self, model_name: str) -> bool:
        """Set the active model."""
        if model_name in self.models:
            self.active_model = model_name
            logger.info(f"Active model set to: {model_name}")
            return True
        return False
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

# Initialize FastAPI app and model store
app = FastAPI(
    title="Pumpkin Price Prediction API",
    description="A machine learning API for predicting pumpkin prices using various models",
    version="1.0.0",
    contact={
        "name": "PyData 2025 Bazel AI Tutorial",
        "email": "example@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

model_store = ModelStore()

def process_features(features: PumpkinFeatures, model_metadata: Dict) -> pd.DataFrame:
    """Process input features into the format expected by the model."""
    feature_names = model_metadata.get('feature_names', [])
    
    # Create base features
    processed_features = {
        'DayOfYear': features.day_of_year,
        'Month': features.month,
    }
    
    # Handle one-hot encoded categorical features
    for feature_name in feature_names:
        if feature_name.startswith('Variety_'):
            variety = feature_name.replace('Variety_', '')
            processed_features[feature_name] = 1 if features.variety == variety else 0
        elif feature_name.startswith('City_'):
            city = feature_name.replace('City_', '')
            processed_features[feature_name] = 1 if features.city == city else 0
        elif feature_name.startswith('Package_'):
            package = feature_name.replace('Package_', '')
            processed_features[feature_name] = 1 if features.package == package else 0
        elif feature_name not in processed_features:
            # Default to 0 for unknown features
            processed_features[feature_name] = 0
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([processed_features])
    
    # Ensure all required features are present
    for feature_name in feature_names:
        if feature_name not in df.columns:
            df[feature_name] = 0
    
    # Reorder columns to match training
    df = df[feature_names]
    
    return df

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting Pumpkin Price Prediction API...")
    
    # Try to load models from default locations
    models_dirs = [
        "bazel-bin/examples/02-basic-ml/ml_pipeline.runfiles/_main/output/models",
        "output/models",
        "models"
    ]
    
    loaded_count = 0
    for models_dir in models_dirs:
        if os.path.exists(models_dir):
            metadata_path = os.path.join(models_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    for model_name, info in metadata.items():
                        model_path = os.path.join(models_dir, info['model_file'])
                        if os.path.exists(model_path):
                            if model_store.load_model(model_path, model_name, info):
                                loaded_count += 1
                    break  # Successfully loaded from this directory
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
    
    if loaded_count == 0:
        logger.warning("No models loaded. Some endpoints may not work.")
    else:
        logger.info(f"Loaded {loaded_count} models successfully")

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Pumpkin Price Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "models": "/models"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if model_store.models else "no_models",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=len(model_store.models),
        active_model=model_store.active_model or "none",
        uptime_seconds=model_store.get_uptime()
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models with their metadata."""
    model_info_list = []
    
    for model_name in model_store.list_models():
        metadata = model_store.metadata[model_name]
        model_info_list.append(
            ModelInfo(
                model_name=model_name,
                model_type=metadata.get('model_type', 'unknown'),
                feature_names=metadata.get('feature_names', []),
                performance_metrics=metadata.get('performance', {}),
                training_data_shape=metadata.get('training_data_shape', 'unknown'),
                last_updated=datetime.now(),  # Would be loaded from metadata in production
                is_active=(model_name == model_store.active_model)
            )
        )
    
    return model_info_list

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str = Path(..., description="Name of the model")):
    """Get detailed information about a specific model."""
    if model_name not in model_store.metadata:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    metadata = model_store.metadata[model_name]
    return ModelInfo(
        model_name=model_name,
        model_type=metadata.get('model_type', 'unknown'),
        feature_names=metadata.get('feature_names', []),
        performance_metrics=metadata.get('performance', {}),
        training_data_shape=metadata.get('training_data_shape', 'unknown'),
        last_updated=datetime.now(),
        is_active=(model_name == model_store.active_model)
    )

@app.post("/models/{model_name}/activate")
async def activate_model(model_name: str = Path(..., description="Name of the model to activate")):
    """Activate a specific model for predictions."""
    if model_store.set_active_model(model_name):
        return {"message": f"Model '{model_name}' activated successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: PumpkinFeatures, model_name: Optional[str] = None):
    """Make a single prediction."""
    start_time = datetime.now()
    
    try:
        model, metadata = model_store.get_model(model_name)
        
        # Process features
        processed_features = process_features(features, metadata)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        
        # Create response
        response = PredictionResponse(
            predicted_price=float(prediction),
            confidence_interval=None,  # Could be implemented with prediction intervals
            model_used=model_name or model_store.active_model,
            prediction_timestamp=start_time,
            features_used=processed_features.iloc[0].to_dict()
        )
        
        logger.info(f"Prediction made: {prediction:.2f} using model {response.model_used}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    start_time = datetime.now()
    
    try:
        model, metadata = model_store.get_model(request.model_name)
        
        predictions = []
        for features in request.samples:
            # Process features
            processed_features = process_features(features, metadata)
            
            # Make prediction
            prediction = model.predict(processed_features)[0]
            
            # Create response
            pred_response = PredictionResponse(
                predicted_price=float(prediction),
                confidence_interval=None,
                model_used=request.model_name or model_store.active_model,
                prediction_timestamp=datetime.now(),
                features_used=processed_features.iloc[0].to_dict()
            )
            predictions.append(pred_response)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            processing_time_ms=processing_time,
            model_used=request.model_name or model_store.active_model
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} samples in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/predict/sample")
async def get_sample_prediction():
    """Get a sample prediction using default values."""
    sample_features = PumpkinFeatures(
        day_of_year=280,
        month=10,
        variety="PIE TYPE",
        city="BOSTON",
        package="bushel cartons"
    )
    
    return await predict_single(sample_features)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Pumpkin Price Prediction API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "model_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )