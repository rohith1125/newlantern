import logging
import time
import hashlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predictor import parse_study, predict_relevance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

_cache: dict[str, bool] = {}


def _pair_key(current_desc: str, prior_desc: str) -> str:
    raw = f"{current_desc.upper()}||{prior_desc.upper()}"
    return hashlib.md5(raw.encode()).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("server ready")
    yield
    log.info("server shutting down")


app = FastAPI(title="relevant-priors", lifespan=lifespan)


class StudyIn(BaseModel):
    study_id: str
    study_description: str
    study_date: str | None = None


class CaseIn(BaseModel):
    case_id: str
    patient_id: str | None = None
    patient_name: str | None = None
    current_study: StudyIn
    prior_studies: list[StudyIn]


class PredictionRequest(BaseModel):
    challenge_id: str | None = None
    schema_version: int | None = None
    generated_at: str | None = None
    cases: list[CaseIn]


class PredictionOut(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictionResponse(BaseModel):
    predictions: list[PredictionOut]


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, req: Request) -> Any:
    t0 = time.perf_counter()
    total_priors = sum(len(c.prior_studies) for c in payload.cases)
    log.info(
        "request cases=%d total_priors=%d",
        len(payload.cases),
        total_priors,
    )

    predictions: list[PredictionOut] = []

    for case in payload.cases:
        current = parse_study(
            case.current_study.study_description,
            case.current_study.study_date,
        )

        for prior in case.prior_studies:
            key = _pair_key(
                case.current_study.study_description,
                prior.study_description,
            )

            if key in _cache:
                result = _cache[key]
            else:
                prior_parsed = parse_study(prior.study_description, prior.study_date)
                result = predict_relevance(current, prior_parsed)
                _cache[key] = result

            predictions.append(
                PredictionOut(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=result,
                )
            )

    elapsed = time.perf_counter() - t0
    log.info(
        "done predictions=%d elapsed=%.3fs",
        len(predictions),
        elapsed,
    )

    return PredictionResponse(predictions=predictions)


@app.get("/health")
def health():
    return {"status": "ok", "cache_size": len(_cache)}
