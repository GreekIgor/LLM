"""
FastAPI Inference Service для vLLM с Prometheus метриками и MLflow интеграцией
Основано на паттерне из https://github.com/Ilia2704/llm_mlflow
"""

import os
import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import Response

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import mlflow

# ===== Prometheus Метрики =====
REQ_COUNTER = Counter(
    "llm_requests_total", 
    "Total LLM requests", 
    ["endpoint", "model"]
)

TOKENS_COUNTER = Counter(
    "llm_tokens_total", 
    "Total tokens processed", 
    ["phase", "model"]  # phase: input|output
)

LATENCY_HIST = Histogram(
    "llm_request_latency_seconds", 
    "LLM request latency in seconds",
    ["endpoint", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)
)

ERROR_COUNTER = Counter(
    "llm_errors_total",
    "Total LLM errors",
    ["endpoint", "error_type"]
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Number of active requests"
)

# ===== Pydantic Models =====
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling probability")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    model: Optional[str] = Field(None, description="Model name (optional)")

class GenerateResponse(BaseModel):
    output: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    timestamp: float

class HealthResponse(BaseModel):
    status: str
    vllm_url: str
    mlflow_tracking_uri: str
    timestamp: float

# ===== FastAPI Application =====
app = FastAPI(
    title="vLLM Inference Service",
    description="Production-ready LLM inference with MLflow tracking and Prometheus metrics",
    version="1.0.0"
)

# ===== Configuration =====
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "vllm-inference")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "facebook/opt-1.3b")

# Setup MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
except Exception as e:
    print(f"Warning: Could not set MLflow experiment: {e}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "vLLM Inference Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "generate": "/generate",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        vllm_url=VLLM_BASE_URL,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        timestamp=time.time()
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint"""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(req: GenerateRequest):
    """
    Generate text using vLLM backend
    
    Logs metrics to Prometheus and experiment data to MLflow
    """
    model_name = req.model or DEFAULT_MODEL
    
    # Update active requests gauge
    ACTIVE_REQUESTS.inc()
    
    # Start timing
    start_time = time.perf_counter()
    
    # Increment request counter
    REQ_COUNTER.labels(endpoint="/generate", model=model_name).inc()
    
    try:
        # Import here to avoid startup dependency
        import requests
        
        # Start MLflow run
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params({
                "model": model_name,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "prompt_length": len(req.prompt),
            })
            
            # Estimate input tokens (rough approximation)
            input_tokens = len(req.prompt.split())
            TOKENS_COUNTER.labels(phase="input", model=model_name).inc(input_tokens)
            
            # Call vLLM API
            payload = {
                "model": model_name,
                "prompt": req.prompt,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
            }
            
            if req.stop:
                payload["stop"] = req.stop
            
            response = requests.post(
                f"{VLLM_BASE_URL}/v1/completions",
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract results
            output_text = result["choices"][0]["text"]
            usage = result.get("usage", {})
            
            # Get actual token counts from vLLM response
            actual_input_tokens = usage.get("prompt_tokens", input_tokens)
            actual_output_tokens = usage.get("completion_tokens", len(output_text.split()))
            total_tokens = usage.get("total_tokens", actual_input_tokens + actual_output_tokens)
            
            # Update output tokens counter
            TOKENS_COUNTER.labels(phase="output", model=model_name).inc(actual_output_tokens)
            
            # Calculate latency
            latency = time.perf_counter() - start_time
            LATENCY_HIST.labels(endpoint="/generate", model=model_name).observe(latency)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "latency_seconds": latency,
                "input_tokens": actual_input_tokens,
                "output_tokens": actual_output_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": actual_output_tokens / latency if latency > 0 else 0,
            })
            
            # Log prompt and output as artifacts (optional, for debugging)
            if os.getenv("LOG_PROMPTS", "false").lower() == "true":
                mlflow.log_text(req.prompt, "prompt.txt")
                mlflow.log_text(output_text, "output.txt")
            
            return GenerateResponse(
                output=output_text,
                model=model_name,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                timestamp=time.time()
            )
            
    except requests.exceptions.ConnectionError:
        ERROR_COUNTER.labels(endpoint="/generate", error_type="connection_error").inc()
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to vLLM server at {VLLM_BASE_URL}"
        )
    except requests.exceptions.Timeout:
        ERROR_COUNTER.labels(endpoint="/generate", error_type="timeout").inc()
        raise HTTPException(
            status_code=504,
            detail="Request to vLLM server timed out"
        )
    except requests.exceptions.HTTPError as e:
        ERROR_COUNTER.labels(endpoint="/generate", error_type="http_error").inc()
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"vLLM server error: {e.response.text}"
        )
    except Exception as e:
        ERROR_COUNTER.labels(endpoint="/generate", error_type="unknown").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Decrement active requests
        ACTIVE_REQUESTS.dec()


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"""
    🚀 Starting vLLM Inference Service
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📍 Server: http://{host}:{port}
    📊 Metrics: http://{host}:{port}/metrics
    📖 Docs: http://{host}:{port}/docs
    🔌 vLLM Backend: {VLLM_BASE_URL}
    📈 MLflow: {MLFLOW_TRACKING_URI}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
