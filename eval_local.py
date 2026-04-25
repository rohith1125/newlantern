"""
Quick local accuracy check against the downloaded public eval JSON.

Usage:
    python eval_local.py public_eval.json
    python eval_local.py public_eval.json --endpoint http://localhost:8000/predict
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

from predictor import parse_study, predict_relevance


def run_local(cases):
    predictions = []
    for case in cases:
        current = parse_study(
            case["current_study"]["study_description"],
            case["current_study"].get("study_date"),
        )
        for prior in case["prior_studies"]:
            prior_parsed = parse_study(
                prior["study_description"],
                prior.get("study_date"),
            )
            result = predict_relevance(current, prior_parsed)
            predictions.append({
                "case_id": case["case_id"],
                "study_id": prior["study_id"],
                "predicted_is_relevant": result,
            })
    return predictions


def run_endpoint(endpoint, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=360) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    print(f"endpoint latency: {elapsed:.2f}s")
    return body["predictions"]


def score(predictions, labeled_cases):
    truth = {}
    for case in labeled_cases:
        for prior in case["prior_studies"]:
            truth[(case["case_id"], prior["study_id"])] = prior["is_relevant"]

    correct = 0
    incorrect = 0
    skipped = 0

    pred_map = {(p["case_id"], p["study_id"]): p["predicted_is_relevant"] for p in predictions}

    for (cid, sid), label in truth.items():
        if (cid, sid) not in pred_map:
            skipped += 1
        elif pred_map[(cid, sid)] == label:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect + skipped
    accuracy = correct / total if total else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "incorrect": incorrect,
        "skipped": skipped,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json")
    parser.add_argument("--endpoint", default=None)
    args = parser.parse_args()

    with open(args.eval_json) as f:
        data = json.load(f)

    cases = data if isinstance(data, list) else data.get("cases", data)

    if args.endpoint:
        payload = {"challenge_id": "relevant-priors-v1", "schema_version": 1, "cases": cases}
        predictions = run_endpoint(args.endpoint, payload)
    else:
        predictions = run_local(cases)

    result = score(predictions, cases)
    print(json.dumps(result, indent=2))
    print(f"\nAccuracy: {result['accuracy']:.4f}  ({result['correct']}/{result['total']})")


if __name__ == "__main__":
    main()
