"""
Health Monitoring Service for ML API.

This service provides:
1. Model performance monitoring
2. System health checks
3. Alert generation
4. Metrics collection and reporting
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """Health metrics data structure."""
    timestamp: datetime
    model_name: str
    prediction_count: int
    avg_response_time_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    predictions_per_minute: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    timestamp: datetime
    severity: str  # "low", "medium", "high", "critical"
    component: str
    message: str
    details: Dict
    resolved: bool = False
    
    def to_dict(self):
        return asdict(self)

class HealthMonitor:
    """Health monitoring system."""
    
    def __init__(self):
        self.metrics_history: List[HealthMetrics] = []
        self.alerts: List[Alert] = []
        self.prediction_log: List[Dict] = []
        self.start_time = datetime.now()
        
        # Thresholds for alerting
        self.thresholds = {
            "max_response_time_ms": 1000,
            "max_error_rate": 0.05,  # 5%
            "max_memory_usage_mb": 1000,
            "max_cpu_usage_percent": 80,
            "min_predictions_per_minute": 1
        }
    
    def log_prediction(self, model_name: str, response_time_ms: float, 
                      success: bool, error_msg: str = None):
        """Log a prediction request."""
        self.prediction_log.append({
            "timestamp": datetime.now(),
            "model_name": model_name,
            "response_time_ms": response_time_ms,
            "success": success,
            "error_msg": error_msg
        })
        
        # Keep only recent logs (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.prediction_log = [log for log in self.prediction_log 
                             if log["timestamp"] > cutoff_time]
    
    def collect_metrics(self, model_name: str) -> HealthMetrics:
        """Collect current health metrics."""
        now = datetime.now()
        
        # Analyze recent predictions (last 5 minutes)
        recent_cutoff = now - timedelta(minutes=5)
        recent_predictions = [log for log in self.prediction_log 
                            if log["timestamp"] > recent_cutoff 
                            and log["model_name"] == model_name]
        
        # Calculate metrics
        prediction_count = len(recent_predictions)
        
        if recent_predictions:
            avg_response_time = np.mean([p["response_time_ms"] for p in recent_predictions])
            error_count = sum(1 for p in recent_predictions if not p["success"])
            error_rate = error_count / prediction_count if prediction_count > 0 else 0
        else:
            avg_response_time = 0
            error_rate = 0
        
        # Calculate predictions per minute
        predictions_per_minute = prediction_count  # Since we're looking at 5 minutes, divide by 5
        if prediction_count > 0:
            predictions_per_minute = prediction_count / 5.0
        
        # System metrics (simplified - in production, use psutil)
        memory_usage_mb = self._get_memory_usage()
        cpu_usage_percent = self._get_cpu_usage()
        
        metrics = HealthMetrics(
            timestamp=now,
            model_name=model_name,
            prediction_count=prediction_count,
            avg_response_time_ms=avg_response_time,
            error_rate=error_rate,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            predictions_per_minute=predictions_per_minute
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 24 hours)
        cutoff_time = now - timedelta(hours=24)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil not available
            return np.random.uniform(100, 500)  # Mock data
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage (simplified)."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            # Fallback if psutil not available
            return np.random.uniform(10, 50)  # Mock data
    
    def _check_alerts(self, metrics: HealthMetrics):
        """Check metrics against thresholds and generate alerts."""
        alerts_to_create = []
        
        # Response time alert
        if metrics.avg_response_time_ms > self.thresholds["max_response_time_ms"]:
            alerts_to_create.append({
                "severity": "medium",
                "component": "response_time",
                "message": f"High response time detected: {metrics.avg_response_time_ms:.2f}ms",
                "details": {"threshold": self.thresholds["max_response_time_ms"], 
                           "actual": metrics.avg_response_time_ms}
            })
        
        # Error rate alert
        if metrics.error_rate > self.thresholds["max_error_rate"]:
            alerts_to_create.append({
                "severity": "high",
                "component": "error_rate",
                "message": f"High error rate detected: {metrics.error_rate*100:.2f}%",
                "details": {"threshold": self.thresholds["max_error_rate"]*100, 
                           "actual": metrics.error_rate*100}
            })
        
        # Memory usage alert
        if metrics.memory_usage_mb > self.thresholds["max_memory_usage_mb"]:
            alerts_to_create.append({
                "severity": "medium",
                "component": "memory",
                "message": f"High memory usage: {metrics.memory_usage_mb:.2f}MB",
                "details": {"threshold": self.thresholds["max_memory_usage_mb"], 
                           "actual": metrics.memory_usage_mb}
            })
        
        # CPU usage alert
        if metrics.cpu_usage_percent > self.thresholds["max_cpu_usage_percent"]:
            alerts_to_create.append({
                "severity": "medium",
                "component": "cpu",
                "message": f"High CPU usage: {metrics.cpu_usage_percent:.2f}%",
                "details": {"threshold": self.thresholds["max_cpu_usage_percent"], 
                           "actual": metrics.cpu_usage_percent}
            })
        
        # Low activity alert
        if metrics.predictions_per_minute < self.thresholds["min_predictions_per_minute"]:
            alerts_to_create.append({
                "severity": "low",
                "component": "activity",
                "message": f"Low prediction activity: {metrics.predictions_per_minute:.2f}/min",
                "details": {"threshold": self.thresholds["min_predictions_per_minute"], 
                           "actual": metrics.predictions_per_minute}
            })
        
        # Create alerts
        for alert_data in alerts_to_create:
            alert = Alert(
                alert_id=f"{alert_data['component']}_{int(time.time())}",
                timestamp=metrics.timestamp,
                severity=alert_data["severity"],
                component=alert_data["component"],
                message=alert_data["message"],
                details=alert_data["details"]
            )
            self.alerts.append(alert)
            logger.warning(f"Alert generated: {alert.message}")
    
    def get_metrics_summary(self, model_name: str = None, hours: int = 1) -> Dict:
        """Get metrics summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if model_name:
            relevant_metrics = [m for m in self.metrics_history 
                              if m.timestamp > cutoff_time and m.model_name == model_name]
        else:
            relevant_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not relevant_metrics:
            return {"error": "No metrics available for the specified period"}
        
        # Calculate summary statistics
        response_times = [m.avg_response_time_ms for m in relevant_metrics]
        error_rates = [m.error_rate for m in relevant_metrics]
        memory_usage = [m.memory_usage_mb for m in relevant_metrics]
        cpu_usage = [m.cpu_usage_percent for m in relevant_metrics]
        
        return {
            "time_period_hours": hours,
            "model_name": model_name or "all",
            "metrics_count": len(relevant_metrics),
            "response_time": {
                "avg_ms": np.mean(response_times),
                "max_ms": np.max(response_times),
                "min_ms": np.min(response_times),
                "p95_ms": np.percentile(response_times, 95)
            },
            "error_rate": {
                "avg_percent": np.mean(error_rates) * 100,
                "max_percent": np.max(error_rates) * 100
            },
            "resource_usage": {
                "avg_memory_mb": np.mean(memory_usage),
                "max_memory_mb": np.max(memory_usage),
                "avg_cpu_percent": np.mean(cpu_usage),
                "max_cpu_percent": np.max(cpu_usage)
            },
            "total_predictions": sum(m.prediction_count for m in relevant_metrics)
        }
    
    def get_active_alerts(self, severity: str = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if severity:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_uptime(self) -> Dict:
        """Get system uptime information."""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600,
            "uptime_days": uptime_seconds / 86400
        }

# FastAPI app for health monitoring
monitor_app = FastAPI(
    title="ML Model Health Monitor",
    description="Health monitoring and alerting for ML model API",
    version="1.0.0"
)

# Global health monitor instance
health_monitor = HealthMonitor()

class MetricsResponse(BaseModel):
    """Response model for metrics."""
    timestamp: datetime
    model_name: str
    prediction_count: int
    avg_response_time_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    predictions_per_minute: float

class AlertResponse(BaseModel):
    """Response model for alerts."""
    alert_id: str
    timestamp: datetime
    severity: str
    component: str
    message: str
    details: Dict
    resolved: bool

@monitor_app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ML Model Health Monitor",
        "version": "1.0.0",
        "endpoints": {
            "metrics": "/metrics/{model_name}",
            "alerts": "/alerts",
            "summary": "/summary",
            "uptime": "/uptime"
        }
    }

@monitor_app.get("/metrics/{model_name}", response_model=MetricsResponse)
async def get_current_metrics(model_name: str):
    """Get current metrics for a specific model."""
    metrics = health_monitor.collect_metrics(model_name)
    return MetricsResponse(**metrics.to_dict())

@monitor_app.get("/summary")
async def get_metrics_summary(model_name: str = None, hours: int = 1):
    """Get metrics summary."""
    return health_monitor.get_metrics_summary(model_name, hours)

@monitor_app.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(severity: str = None, active_only: bool = True):
    """Get alerts."""
    if active_only:
        alerts = health_monitor.get_active_alerts(severity)
    else:
        alerts = health_monitor.alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
    
    return [AlertResponse(**alert.to_dict()) for alert in alerts]

@monitor_app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    success = health_monitor.resolve_alert(alert_id)
    return {"success": success, "alert_id": alert_id}

@monitor_app.get("/uptime")
async def get_uptime():
    """Get system uptime."""
    return health_monitor.get_uptime()

@monitor_app.post("/log-prediction")
async def log_prediction(model_name: str, response_time_ms: float, 
                        success: bool, error_msg: str = None):
    """Log a prediction for monitoring."""
    health_monitor.log_prediction(model_name, response_time_ms, success, error_msg)
    return {"status": "logged"}

async def continuous_monitoring():
    """Background task for continuous monitoring."""
    while True:
        try:
            # This would normally monitor all active models
            # For demo, we'll just collect metrics for a default model
            health_monitor.collect_metrics("default_model")
            await asyncio.sleep(60)  # Collect metrics every minute
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {e}")
            await asyncio.sleep(60)

@monitor_app.on_event("startup")
async def startup_event():
    """Start background monitoring tasks."""
    logger.info("Starting health monitoring service...")
    # Start background monitoring task
    asyncio.create_task(continuous_monitoring())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Health Monitor API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "health_monitor:monitor_app",
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )