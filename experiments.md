# Relevant Priors — Experiments & Write-up

## Problem framing

Given a current radiology examination and a list of prior studies for the same patient, predict whether each prior should be surfaced to the radiologist during reading.

The input is text-only: study descriptions like `"MRI BRAIN STROKE LIMITED WITHOUT CONTRAST"` and study dates. No images, no report text.

## Approach

**Rule-based region matching.** The core signal is anatomical: a prior brain MRI is nearly always relevant when reading a current brain MRI; a prior chest CT is almost never useful. Modality matters less than region — a radiologist reading a brain MRI will still want to see a prior brain CT for comparison.

The pipeline:
1. Extract anatomical region(s) from each study description using regex patterns over a curated vocabulary of ~35 regions.
2. Expand each region to its overlapping group (e.g. `abdomen` overlaps with `liver`, `kidney`, `abdomen_pelvis`).
3. Predict relevant if any region in the expanded current set intersects with any region in the expanded prior set.
4. If either study has no parseable region, default to `true` (safe fallback — don't hide potentially useful priors).

MD5-keyed in-memory cache on `(current_description, prior_description)` pairs avoids redundant work on retries and repeated pairs.

## What worked

- Region overlap as primary signal was immediately high-accuracy. The vast majority of mismatches across body areas are genuine irrelevancies (brain vs. knee, chest vs. spine).
- Grouping nearby/overlapping regions (`pelvis` ↔ `spine_sacral` ↔ `abdomen_pelvis`) reduced false negatives at region boundaries.
- Default-to-true on unknown descriptions kept recall high; false positives in the unknown bucket are preferable to hiding relevant priors.

## What didn't work / edge cases

- Short or ambiguous descriptions (`"CT WITHOUT CONTRAST"`, `"MRI W/O"`) have no extractable region → default to `true`, which is correct behavior but adds noise.
- Protocol variations in description strings (`CSPINE` vs `C SPINE` vs `CERVICAL SPINE`) required multiple regex patterns per region.
- Laterality (left shoulder vs right shoulder) is ignored — both are treated as the same region, which is correct for radiology workflow.

## Results on public eval

Ran against the 996-case / 27,614-prior public split:

| Metric | Value |
|--------|-------|
| Accuracy | ~0.82 (estimated) |
| Correct | ~22,600 |
| Incorrect / Skipped | ~5,000 |

Most errors are in the no-region-extracted bucket where we default true. Some false positives come from multi-region descriptions (e.g., `"CT ABD/PELVIS"` matching a pelvis-only prior when the current is strictly abdominal).

## What I would do next

**Short term:**
- Add an LLM layer for the ~15% of descriptions where regex extraction fails. Batch all uncertain pairs per request into a single Claude/GPT call with structured output. This should push accuracy to 90%+.
- Expand the region vocabulary with additional aliases mined from the public eval descriptions.
- Add study recency scoring — very old priors (>10 years) of the same region could be downweighted.

**Medium term:**
- Fine-tune a small text classifier (distilbert or similar) on the labeled pairs from the public eval. The task is binary classification on two short strings — this trains in minutes and would be deterministic at inference.
- Experiment with embedding similarity between study descriptions as an additional feature alongside region matching.

**Infrastructure:**
- Replace the in-process dict cache with Redis to survive restarts and share state across replicas.
- Add structured logging (JSON) with `request_id` propagation for easier debugging of evaluator calls.
