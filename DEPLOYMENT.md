# Deployment and Runtime Modes

## Modes

The server has two runtime modes controlled by the `ANTHROPIC_API_KEY` environment variable.

**Rule-based mode (default, no API key needed)**

This is what was used for the submitted endpoint. Predictions are made entirely by the regex-based region extractor in `predictor.py`. No external calls, sub-millisecond latency per prediction.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**LLM mode (requires Anthropic API key)**

When `ANTHROPIC_API_KEY` is set, the server calls Claude Haiku for each case in the request. All priors for a case are batched into one API call. Results are cached by MD5(current_desc + prior_desc) so retries don't re-call the API.

```bash
ANTHROPIC_API_KEY=sk-ant-... uvicorn main:app --host 0.0.0.0 --port 8000
```

The `/health` endpoint reports the active mode:

```json
{"status": "ok", "mode": "rule-based", "cache_size": 0}
```

## Reproducing the submitted endpoint locally

The submission used rule-based mode. To reproduce:

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Then verify:

```bash
curl -s http://localhost:8000/health
# {"status":"ok","mode":"rule-based","cache_size":0}
```

Run accuracy check against the public eval split:

```bash
python eval_local.py relevant_priors_public.json
# Accuracy: 0.9534  (26326/27614)
```

Run regression tests:

```bash
python tests.py
# 44/44 passed - ALL PASS
```

## Docker

```bash
docker build -t relevant-priors .
docker run -p 8000:8000 relevant-priors
# rule-based mode

docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8000:8000 relevant-priors
# LLM mode
```
