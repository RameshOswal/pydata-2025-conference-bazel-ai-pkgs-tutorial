# Model Serving with FastAPI and OpenAPI

This example demonstrates how to serve machine learning models using FastAPI with full OpenAPI documentation, completing the end-to-end ML pipeline from training → evaluation → serving.

## 🚀 Features

- **FastAPI Web Framework** - Modern, fast API with automatic OpenAPI/Swagger docs
- **Pydantic Validation** - Request/response validation with type safety
- **Model Management** - Automatic model loading, versioning, and deployment
- **Health Monitoring** - Real-time metrics collection and health checks
- **Batch Predictions** - Support for both single and batch inference
- **Client Library** - Python client for easy API integration
- **Comprehensive Testing** - Full test suite with mocking and integration tests

## 📁 Files Structure

```
04-serve-model/
├── model_server.py      # Main FastAPI application with prediction endpoints
├── model_manager.py     # Model loading, registration, and deployment utilities
├── health_monitor.py    # Health monitoring, metrics collection, and alerting
├── api_client.py        # Python client library for consuming the API
├── test_model_server.py # Comprehensive test suite with unit and integration tests
├── demo.py             # Complete pipeline demonstration script
├── BUILD               # Bazel build configuration
└── README.md           # This documentation
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   02-basic-ml   │───▶│03-model-evaluation│───▶│ 04-serve-model  │
│   (Training)    │    │   (Evaluation)    │    │   (Serving)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
   Saves models           Validates models        Serves via API
   with metadata          & generates reports     with health monitoring
```

## 🔧 Installation & Setup

### Prerequisites

This example requires models trained by `02-basic-ml`. The complete pipeline workflow:

1. **Train Models**: Run `02-basic-ml` to train and save models
2. **Evaluate Models**: Run `03-model-evaluation` to validate performance  
3. **Serve Models**: Run `04-serve-model` to deploy via REST API

### Dependencies

The server requires these Python packages (managed by Bazel):
- `fastapi>=0.104.1` - Web framework
- `uvicorn>=0.24.0` - ASGI server  
- `pydantic>=2.5.0` - Data validation
- `pandas`, `numpy`, `scikit-learn` - ML libraries
- `joblib` - Model serialization
- `httpx`, `requests` - HTTP clients for testing

## 🚀 Usage

### Running the Server

```bash
# Using Bazel (recommended)
cd /path/to/pydata-2025-conference-bazel-ai-pkgs-tutorial
bazel run //examples/04-serve-model:model_server

# The server starts on http://localhost:8000
```

### Available Endpoints

Once running, the server provides these endpoints:

- **`GET /`** - API information and status
- **`GET /health`** - Detailed health check with metrics
- **`GET /models`** - List all available models with metadata
- **`POST /predict`** - Single prediction from features
- **`POST /predict/batch`** - Batch predictions (up to 1000 samples)

### Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`  
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 💻 API Examples

### Single Prediction

```python
import requests

# Make a prediction request
response = requests.post("http://localhost:8000/predict", json={
    "day_of_year": 280,
    "month": 10,
    "variety": "PIE TYPE",
    "city": "BOSTON", 
    "package": "bushel cartons"
})

result = response.json()
print(f"Predicted price: ${result['predicted_price']:.2f}")
print(f"Model used: {result['model_used']}")
print(f"Confidence: {result.get('confidence_interval', 'N/A')}")
```

### Batch Predictions

```python
import requests

# Batch prediction request
batch_data = {
    "samples": [
        {
            "day_of_year": 280,
            "month": 10,
            "variety": "PIE TYPE",
            "city": "BOSTON",
            "package": "bushel cartons"
        },
        {
            "day_of_year": 300,
            "month": 11,
            "variety": "CARVING TYPE",
            "city": "NEW YORK",
            "package": "bins"
        }
    ],
    "model_name": null  # Use default/best model
}

response = requests.post("http://localhost:8000/predict/batch", json=batch_data)
results = response.json()

print(f"Processed {results['batch_size']} predictions")
print(f"Processing time: {results['processing_time_ms']:.2f}ms")

for i, pred in enumerate(results['predictions']):
    print(f"Sample {i+1}: ${pred['predicted_price']:.2f}")
```

### Using the Python Client Library

```python
from api_client import APIClient, ClientFeatures

# Initialize client
client = APIClient("http://localhost:8000")

# Check API health
health = client.health_check()
print(f"API Status: {health['status']}")

# Create feature object
features = ClientFeatures(
    day_of_year=280,
    month=10,
    variety="PIE TYPE",
    city="BOSTON",
    package="bushel cartons"
)

# Make prediction
result = client.predict_single(features)
print(f"Predicted price: ${result['predicted_price']:.2f}")

# Validate features
if client.validate_features(features):
    print("✅ Features are valid")
else:
    print("❌ Invalid features")
```

## 🧪 Testing

The package includes comprehensive tests covering all components:

```bash
# Run all tests
bazel test //examples/04-serve-model:test_model_server

# The test suite includes:
# - FastAPI endpoint testing with TestClient
# - Model manager functionality
# - Health monitoring and metrics
# - API client library validation  
# - Error handling and edge cases
# - Integration with trained models
```

## 📊 Health Monitoring

The server includes built-in health monitoring:

```python
import requests

# Get health status
health = requests.get("http://localhost:8000/health").json()

print(f"Status: {health['status']}")
print(f"Uptime: {health['uptime_seconds']}s")
print(f"Memory usage: {health['memory_usage_mb']:.1f}MB")
print(f"Active models: {health['active_models']}")
print(f"Total predictions: {health['prediction_count']}")
```

The health monitor tracks:
- **System Metrics**: Memory usage, uptime, CPU load
- **Model Metrics**: Loaded models, prediction counts, response times
- **Error Tracking**: Failed predictions, error rates, timeouts
- **Performance**: Average response time, throughput, success rates

## 🔄 Complete Pipeline Demo

Run the complete end-to-end pipeline demonstration:

```bash
cd examples/04-serve-model
python3 demo.py
```

This script demonstrates:
1. **Training**: Load data and train models with `02-basic-ml`
2. **Evaluation**: Validate models with `03-model-evaluation`  
3. **Serving**: Configure API server with `04-serve-model`

## 🏭 Production Deployment

For production deployment, consider:

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

```bash
# Environment variables for production
export MODEL_PATH="/data/models"
export LOG_LEVEL="INFO"
export MAX_BATCH_SIZE="1000"
export ENABLE_METRICS="true"
export CORS_ORIGINS="https://yourdomain.com"
```

## 🤝 Integration with Pipeline

This serving component integrates seamlessly with the other pipeline stages:

### Model Integration Flow

1. **02-basic-ml** saves trained models using `joblib.dump()`
2. **03-model-evaluation** loads and validates model performance
3. **04-serve-model** automatically discovers and serves the best models

### Supported Model Types

The server supports any scikit-learn compatible model:
- Linear Regression
- Random Forest  
- Support Vector Machines
- Gradient Boosting
- Neural Networks (MLPRegressor)
- Custom ensemble models

### Automatic Model Discovery

The server automatically loads models from these locations:
- `./models/` - Default model directory
- `../02-basic-ml/saved_models/` - Training pipeline output
- `../03-model-evaluation/validated_models/` - Evaluation pipeline output

## 🐛 Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies are installed via Bazel
```bash
bazel test //examples/04-serve-model:test_model_server
```

**No Models Found**: Train models first with `02-basic-ml`
```bash
bazel run //examples/02-basic-ml:ml_pipeline  
```

**Port Already in Use**: Change the port in model_server.py
```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port
```

**Pydantic Warnings**: The server includes config to suppress model namespace warnings

## 📚 API Reference

### Request/Response Models

#### PumpkinFeatures
```python
{
    "day_of_year": int,      # 1-366
    "month": int,            # 1-12  
    "variety": str,          # Pumpkin variety
    "city": str,            # City name
    "package": str          # Package type
}
```

#### PredictionResponse
```python
{
    "predicted_price": float,           # Predicted price in USD
    "confidence_interval": [float],     # 95% confidence interval (optional)
    "model_used": str,                  # Name of model used
    "prediction_timestamp": datetime,    # When prediction was made
    "features_used": dict               # Processed features used
}
```

#### HealthCheck
```python
{
    "status": str,              # "healthy" or "unhealthy"
    "timestamp": datetime,      # Current timestamp
    "version": str,            # API version
    "uptime_seconds": float,   # Server uptime
    "memory_usage_mb": float,  # Memory usage
    "active_models": int,      # Number of loaded models
    "prediction_count": int    # Total predictions made
}
```

## 🎯 Next Steps

After completing this example, you can:

1. **Extend the API** - Add authentication, rate limiting, caching
2. **Model Versioning** - Implement A/B testing and model rollbacks  
3. **Monitoring** - Add Prometheus metrics and Grafana dashboards
4. **CI/CD** - Automate testing, deployment, and model updates
5. **Documentation** - Generate client SDKs for different languages

This completes the end-to-end ML pipeline from data processing through model serving! 🎉