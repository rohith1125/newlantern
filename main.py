import logging
import os
import time
import hashlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

if USE_LLM:
    from llm_predictor import predict_case_llm, get_cache as llm_get_cache
    log.info("LLM mode: claude-haiku")
else:
    from predictor import parse_study, predict_relevance
    import hashlib as _hm
    _rule_cache: dict[str, bool] = {}
    log.info("rule-based mode (set ANTHROPIC_API_KEY to enable LLM)")


def _rule_pair_key(a: str, b: str) -> str:
    return hashlib.md5(f"{a.upper()}||{b.upper()}".encode()).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("server ready — mode=%s", "llm" if USE_LLM else "rule-based")
    yield


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
    log.info("request cases=%d total_priors=%d", len(payload.cases), total_priors)

    predictions: list[PredictionOut] = []

    for case in payload.cases:
        cur = case.current_study

        if USE_LLM:
            priors_raw = [
                {
                    "study_id": p.study_id,
                    "study_description": p.study_description,
                    "study_date": p.study_date,
                }
                for p in case.prior_studies
            ]
            results = predict_case_llm(
                cur.study_description,
                cur.study_date,
                priors_raw,
            )
            for prior in case.prior_studies:
                predictions.append(
                    PredictionOut(
                        case_id=case.case_id,
                        study_id=prior.study_id,
                        predicted_is_relevant=results.get(prior.study_id, True),
                    )
                )
        else:
            current_parsed = parse_study(cur.study_description, cur.study_date)
            for prior in case.prior_studies:
                key = _rule_pair_key(cur.study_description, prior.study_description)
                if key in _rule_cache:
                    result = _rule_cache[key]
                else:
                    prior_parsed = parse_study(prior.study_description, prior.study_date)
                    result = predict_relevance(current_parsed, prior_parsed)
                    _rule_cache[key] = result
                predictions.append(
                    PredictionOut(
                        case_id=case.case_id,
                        study_id=prior.study_id,
                        predicted_is_relevant=result,
                    )
                )

    elapsed = time.perf_counter() - t0
    log.info("done predictions=%d elapsed=%.3fs", len(predictions), elapsed)

    return PredictionResponse(predictions=predictions)


@app.get("/health")
def health():
    cache_size = len(llm_get_cache()) if USE_LLM else len(_rule_cache)
    return {"status": "ok", "mode": "llm" if USE_LLM else "rule-based", "cache_size": cache_size}
