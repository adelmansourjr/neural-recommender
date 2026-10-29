"""Interactive local API; never substitutes an untrained model."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from .diversity import CANDIDATE_POOL
from .http_guard import ResearchHTTPGuard
from .service import RecommenderService


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    movie_ids: list[StrictInt] = Field(min_length=1, max_length=20)
    k: StrictInt = Field(default=10, ge=1, le=50)
    relevance_weight: StrictFloat = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    mode: Literal[
        "accuracy",
        "adaptive",
        "discovery",
        "blend",
        "hybrid",
        "neural",
        "ease",
        "item_knn",
        "popularity",
        "als",
    ] = "accuracy"


def create_app(
    data_path: Path = Path("data/ml-1m.zip"), artifact_dir: Path = Path("artifacts")
) -> FastAPI:
    app = FastAPI(title="History Neural Recommender", version="0.1.0")
    app.add_middleware(ResearchHTTPGuard)
    service = RecommenderService(data_path, artifact_dir)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    model_name = (
        "history_conditional_selector"
        if service.format_version == 3
        else service.metadata.get("accuracy_model", "hybrid")
        if service.format_version == 2
        else "trained_history_encoder"
    )

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "movies": len(service.catalog.ids),
            "model": model_name,
            "format_version": service.format_version,
        }

    @app.get("/model-info")
    def model_info() -> dict:
        return {
            "format_version": service.format_version,
            "seeds": service.metadata.get("seeds", [service.metadata["config"].get("seed")]),
            "fusion_weights_ease_neural_popularity": service.metadata.get("fusion_weights"),
            "discovery_penalty": service.metadata.get("discovery_penalty"),
            "accuracy_model": model_name,
            "history_selector": service.metadata.get("history_selector"),
            "history_blends": service.metadata.get("history_blends"),
            "selector_rule": service.metadata.get("selector_rule"),
            "als_config": service.metadata.get("als_config"),
            "max_history": service.max_history,
            "note": (
                "Models and history-specific weights are selected on validation, "
                "not reserve test users."
            ),
        }

    @app.get("/movies")
    def movies(q: str = Query(default="", max_length=100)) -> list[dict]:
        return service.search(q)

    @app.post("/recommend")
    def recommend(request: RecommendationRequest) -> dict:
        try:
            selection = service.selection(request.mode, len(request.movie_ids))
            return {
                "recommendations": service.recommend(
                    request.movie_ids, request.k, request.mode, request.relevance_weight
                ),
                "mode": request.mode,
                "selected_ranker": selection["selected_ranker"],
                "selector_history_bucket": selection["history_bucket"],
                "score_type": "Ranking score, not click probability",
                "reranking": {
                    "enabled": request.relevance_weight < 1,
                    "method": "genre_jaccard_mmr",
                    "relevance_weight": request.relevance_weight,
                    "candidate_pool": CANDIDATE_POOL if request.relevance_weight < 1 else None,
                    "score_note": (
                        "Returned scores are original relevance scores; MMR changes order."
                    ),
                },
            }
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return app
