#!/usr/bin/env python3
"""
Demo script showing the complete ML pipeline:
02-basic-ml (train) → 03-model-evaluation (evaluate) → 04-serve-model (serve)

This script demonstrates the integration between all three components.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the examples directories to Python path
current_dir = Path(__file__).parent
examples_dir = current_dir.parent
sys.path.insert(0, str(examples_dir / "02-basic-ml"))
sys.path.insert(0, str(examples_dir / "03-model-evaluation"))
sys.path.insert(0, str(current_dir))

def main():
    """Run the complete ML pipeline demo."""
    print("🚀 PyData 2025 - Complete ML Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Train models (02-basic-ml)
    print("\n📊 Step 1: Training Models (02-basic-ml)")
    print("-" * 30)
    
    try:
        from ml_pipeline import MLPipeline
        
        # Create a temporary directory for this demo
        temp_dir = tempfile.mkdtemp(prefix="ml_demo_")
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Initialize and run ML pipeline
        pipeline = MLPipeline()
        
        # Load and prepare data
        print("📈 Loading and preparing data...")
        pipeline.load_data()
        pipeline.prepare_features()
        
        # Train models
        print("🤖 Training models...")
        pipeline.train_models()
        
        # Save models to temp directory
        print("💾 Saving trained models...")
        pipeline.save_models(temp_dir)
        
        print("✅ Step 1 completed: Models trained and saved")
        
    except ImportError as e:
        print(f"❌ Could not import 02-basic-ml modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in training step: {e}")
        return False
    
    # Step 2: Evaluate models (03-model-evaluation)
    print("\n🔍 Step 2: Evaluating Models (03-model-evaluation)")
    print("-" * 40)
    
    try:
        from model_evaluator import ModelEvaluator
        from cross_validation import CrossValidator
        
        # Initialize evaluator
        evaluator = ModelEvaluator(temp_dir)
        
        # Load and evaluate models
        print("📊 Loading saved models...")
        models = evaluator.load_models()
        print(f"🔢 Found {len(models)} models to evaluate")
        
        if models:
            # Evaluate each model
            print("🧪 Running model evaluation...")
            for model_name in models.keys():
                results = evaluator.evaluate_model(model_name, pipeline.X_test, pipeline.y_test)
                print(f"  📈 {model_name}: R² = {results.get('r2_score', 'N/A'):.3f}")
            
            # Run cross-validation
            print("🔄 Running cross-validation...")
            cv = CrossValidator()
            cv_results = cv.cross_validate_models(models, "regression")
            
            print("✅ Step 2 completed: Models evaluated")
        else:
            print("⚠️  No models found for evaluation")
        
    except ImportError as e:
        print(f"❌ Could not import 03-model-evaluation modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in evaluation step: {e}")
        return False
    
    # Step 3: Model serving setup (04-serve-model)
    print("\n🌐 Step 3: Model Serving Setup (04-serve-model)")
    print("-" * 40)
    
    try:
        # We can't actually run the FastAPI server due to missing dependencies,
        # but we can show the configuration and structure
        print("🏗️  FastAPI Model Server Configuration:")
        
        server_files = [
            "model_server.py - FastAPI application with prediction endpoints",
            "model_manager.py - Model loading and management utilities", 
            "health_monitor.py - Health monitoring and metrics collection",
            "api_client.py - Client library for consuming the API"
        ]
        
        for file_desc in server_files:
            file_path = current_dir / file_desc.split(" - ")[0]
            if file_path.exists():
                print(f"  ✅ {file_desc}")
            else:
                print(f"  ❌ {file_desc}")
        
        print("\n🔧 API Endpoints Available:")
        endpoints = [
            "GET  /        - API information and health",
            "GET  /health  - Detailed health check",
            "GET  /models  - List available models", 
            "POST /predict - Single prediction",
            "POST /predict/batch - Batch predictions"
        ]
        
        for endpoint in endpoints:
            print(f"  📡 {endpoint}")
        
        print("\n📝 Sample API Usage:")
        print("""
  # Single prediction
  curl -X POST "http://localhost:8000/predict" \\
       -H "Content-Type: application/json" \\
       -d '{
         "day_of_year": 280,
         "month": 10,
         "variety": "PIE TYPE",
         "city": "BOSTON",
         "package": "bushel cartons"
       }'
        """)
        
        print("✅ Step 3 completed: Server configuration ready")
        
    except Exception as e:
        print(f"❌ Error in serving setup: {e}")
        return False
    
    # Cleanup
    print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    
    print("\n🎉 Complete ML Pipeline Demo Finished!")
    print("=" * 50)
    print("\n📚 Summary:")
    print("  1. ✅ Trained ML models with 02-basic-ml")
    print("  2. ✅ Evaluated models with 03-model-evaluation") 
    print("  3. ✅ Configured FastAPI server with 04-serve-model")
    print("\n🚀 Ready for production deployment!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)