"""
LLM-based relevance predictor using Claude.
Batches all prior studies for a case into one API call.
Falls back to rule-based predictor on API errors.
"""

import json
import os
import hashlib
import logging
from typing import Optional

import anthropic

from predictor import parse_study, predict_relevance, StudyFeatures

log = logging.getLogger(__name__)

_client: Optional[anthropic.Anthropic] = None
_cache: dict[str, bool] = {}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _pair_key(current_desc: str, prior_desc: str) -> str:
    raw = f"{current_desc.upper()}||{prior_desc.upper()}"
    return hashlib.md5(raw.encode()).hexdigest()


SYSTEM_PROMPT = """You are a radiologist's workflow assistant. Given a current radiology study and a list of prior studies for the same patient, decide which prior studies are relevant for the radiologist to compare during the current read.

A prior is RELEVANT if it:
- Images the same or overlapping anatomical region as the current study
- Would help the radiologist detect changes, progression, or baseline findings

A prior is NOT RELEVANT if it:
- Images a completely different body region (e.g., a knee study when reading a brain study)
- Would provide no useful comparative information

Respond with ONLY a JSON array. Each element: {"study_id": "...", "relevant": true/false}
No explanation, no markdown, just the raw JSON array."""


def _build_prompt(current_desc: str, current_date: Optional[str], priors: list[dict]) -> str:
    lines = [f'Current study: "{current_desc}"']
    if current_date:
        lines[0] += f" ({current_date})"
    lines.append("")
    lines.append("Prior studies:")
    for p in priors:
        line = f'  - id={p["study_id"]} "{p["study_description"]}"'
        if p.get("study_date"):
            line += f' ({p["study_date"]})'
        lines.append(line)
    lines.append("")
    lines.append("Return JSON array with relevant field for each prior study_id.")
    return "\n".join(lines)


def predict_case_llm(
    current_desc: str,
    current_date: Optional[str],
    priors: list[dict],
) -> dict[str, bool]:
    """
    Returns {study_id: bool} for all priors in the case.
    Checks cache first, only calls API for uncached pairs.
    """
    uncached = [p for p in priors if _pair_key(current_desc, p["study_description"]) not in _cache]

    if uncached:
        prompt = _build_prompt(current_desc, current_date, uncached)
        try:
            client = _get_client()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            results = json.loads(raw)
            for item in results:
                sid = item["study_id"]
                relevant = bool(item["relevant"])
                # find description for this study_id
                prior_map = {p["study_id"]: p["study_description"] for p in uncached}
                if sid in prior_map:
                    key = _pair_key(current_desc, prior_map[sid])
                    _cache[key] = relevant
        except Exception as e:
            log.warning("LLM call failed, falling back to rule-based: %s", e)

        # Any uncached priors the model didn't return get rule-based fallback.
        # Never default to True for missing model output.
        current_feat = parse_study(current_desc, current_date)
        for p in uncached:
            key = _pair_key(current_desc, p["study_description"])
            if key not in _cache:
                prior_feat = parse_study(p["study_description"], p.get("study_date"))
                _cache[key] = predict_relevance(current_feat, prior_feat)
                log.debug("rule-based fallback for study_id=%s", p["study_id"])

    def _get(p: dict) -> bool:
        key = _pair_key(current_desc, p["study_description"])
        if key in _cache:
            return _cache[key]
        # Should not reach here, but rule-based is safer than raising.
        log.warning("cache miss for study_id=%s — applying rule-based", p["study_id"])
        feat = parse_study(p["study_description"], p.get("study_date"))
        result = predict_relevance(parse_study(current_desc, current_date), feat)
        _cache[key] = result
        return result

    return {p["study_id"]: _get(p) for p in priors}


def get_cache() -> dict:
    return _cache
