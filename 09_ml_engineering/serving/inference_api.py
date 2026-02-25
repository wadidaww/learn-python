"""
serving/inference_api.py
=========================
Model serving API using FastAPI (conditional import).

Exposes:
  POST /predict        – single prediction
  POST /predict/batch  – batch predictions
  GET  /model/info     – metadata about the loaded model
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pure-Python model registry (works without FastAPI)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Simple in-memory model registry."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        model: Any,
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        """Register a model under *name*."""
        self._models[name] = model
        self._metadata[name] = {
            "name":        name,
            "version":     version,
            "description": description,
            "registered_at": time.time(),
        }

    def get(self, name: str) -> Any:
        """Return model by *name*; raise KeyError if not found."""
        if name not in self._models:
            raise KeyError(f"No model registered as {name!r}")
        return self._models[name]

    def info(self, name: str) -> dict[str, Any]:
        """Return metadata for *name*."""
        return dict(self._metadata.get(name, {}))

    def list_models(self) -> list[str]:
        return list(self._models.keys())


# Global registry (singleton-style)
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Return the global model registry."""
    return _registry


if FASTAPI_AVAILABLE:
    class PredictRequest(BaseModel):
        features: list[float] = Field(..., description="Feature vector")
        model_name: str = Field("default", description="Registered model name")

    class BatchPredictRequest(BaseModel):
        instances: list[list[float]] = Field(..., description="Batch of feature vectors")
        model_name: str = Field("default")

    class PredictResponse(BaseModel):
        prediction: float
        probability: float | None = None
        model_name: str
        latency_ms: float

    class BatchPredictResponse(BaseModel):
        predictions: list[float]
        model_name: str
        latency_ms: float

    def create_inference_app() -> FastAPI:
        """Create and configure the inference FastAPI app."""
        app = FastAPI(
            title="ML Inference API",
            version="0.1.0",
            description="Serve trained models via REST",
        )

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/model/info")
        async def model_info(model_name: str = "default") -> dict[str, Any]:
            registry = get_registry()
            try:
                return registry.info(model_name)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {model_name!r} not found",
                )

        @app.get("/models")
        async def list_models() -> list[str]:
            return get_registry().list_models()

        @app.post("/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest) -> PredictResponse:
            registry = get_registry()
            try:
                model = registry.get(request.model_name)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {request.model_name!r} not registered",
                )
            start = time.perf_counter()
            X = [request.features]
            preds = model.predict(X)
            prob: float | None = None
            try:
                proba = model.predict_proba(X)
                prob = proba[0][1]
            except NotImplementedError:
                pass
            latency = (time.perf_counter() - start) * 1000
            return PredictResponse(
                prediction=float(preds[0]),
                probability=prob,
                model_name=request.model_name,
                latency_ms=round(latency, 3),
            )

        @app.post("/predict/batch", response_model=BatchPredictResponse)
        async def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
            registry = get_registry()
            try:
                model = registry.get(request.model_name)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {request.model_name!r} not registered",
                )
            start = time.perf_counter()
            preds = model.predict(request.instances)
            latency = (time.perf_counter() - start) * 1000
            return BatchPredictResponse(
                predictions=[float(p) for p in preds],
                model_name=request.model_name,
                latency_ms=round(latency, 3),
            )

        return app

    app = create_inference_app()
