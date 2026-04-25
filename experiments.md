# Relevant Priors: Experiments and Write-up

## Problem Setup

Given a current radiology study and a list of prior studies for the same patient, predict which priors are relevant for the radiologist to compare during reading. The input is text only: study descriptions like "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST" and study dates. No images, no report text.

The public eval has 27,614 labeled prior/current pairs across 996 cases. Only 23.8% of priors are actually relevant, so the baseline of always predicting false gives 76.2% accuracy. The goal was to beat that significantly.

## Approach

The core insight is anatomical: a prior brain MRI is almost always useful when reading a current brain MRI, and a prior knee X-ray is almost never useful. Modality matters less than body region for the relevance decision.

I built a rule-based region extractor that parses study descriptions using regex patterns over a vocabulary of 35+ anatomical regions. For each pair, I expand each study's regions to their anatomically adjacent neighbors, then check for overlap.

Examples of how expansion works:
- spine_lumbar expands to include spine_sacral (adjacent), but not spine_thoracic or spine_cervical
- abdomen_pelvis expands to cover both abdomen and pelvis; but a pure abdomen study does NOT directly match a pure pelvis study
- whole_body (bone scans, PET skull-to-thigh) expands to match most body regions

Default behavior: if either study has no parseable region, predict false. The base rate is low enough (24% relevant) that it is safer to abstain than to guess.

## Iteration Log

**Iteration 1: initial regex with default=True**
Accuracy: 58.2% (worse than always-false baseline)
Problem: 11,274 false positives. Too many descriptions with no region defaulted to true.

**Iteration 2: default=False, added MAM/MAMMO patterns**
Accuracy: 90.2%
Most false positives from unrecognized mammography and echo descriptions defaulting to true were eliminated.

**Iteration 3: separated cardiac from chest**
Accuracy: 91.7%
Echocardiograms and chest X-rays ended up grouped together but are not relevant to each other in practice. Separated them into independent regions.

**Iteration 4: neurovascular region, removed barium from chest**
Accuracy: 93.8%
Carotid ultrasounds were incorrectly matching brain MRIs. Created a separate neurovascular region for angio/doppler/carotid studies. Also fixed Modified Barium Swallow incorrectly matching chest X-rays.

**Iteration 5: breast laterality, spine segment independence**
Accuracy: 94.3%
Discovered 171 false positives from right-side mammography matching left-side mammography. Added laterality detection: RT and LT breast studies only match same-side priors; bilateral matches everything. Also fixed spine thoracic matching lumbar through a shared whole_spine group.

**Iteration 6: pelvis/abdomen separation**
Accuracy: 94.6%
Pelvic MRIs were incorrectly matching abdominal studies. Changed the groups so pelvis and abdomen only connect through explicit abdomen_pelvis studies (CT A/P).

**Iteration 7: expanded pattern vocabulary**
Accuracy: 95.3%
Added 12+ missing description patterns from error analysis:
- XR ribs, sternum, thoracentesis to chest
- CERV abbreviation to cervical spine
- Diagnostic target ultrasound, seed localization, standard screening combo to breast
- Endovaginal, bladder to pelvis
- CT LE, lower extremity to lower extremity group
- MRCP, cholangio to liver
- NMmyo, myocardial perfusion to cardiac
- PET PETCT skullbase patterns to whole_body
- PERITONEAL, drainage to abdomen

## Final Results

Public eval: 95.34% accuracy (26,326 / 27,614)
False positives: 657
False negatives: 631

Remaining errors are mostly cardiac vs chest cross-matches (echo vs CT chest) which require modality-aware logic, unrecognized procedure descriptions like "CT guided FNA" where location is not in the text, and some edge cases in breast screening terminology.

## What I Would Do Next

**Short term:**

Add an LLM layer for the ~5% of ambiguous cases. The hints in the challenge already suggest batching all priors per case into one API call. My plan would be to run the rule-based predictor first, then for cases where both studies have no region extracted, send them to Claude Haiku in one batched call per request. This alone should push accuracy past 97%.

**Medium term:**

Fine-tune a small text classifier on the labeled pairs from the public eval. The task is binary classification on two short strings, which trains in minutes on a T4 GPU. A distilbert model with the two study descriptions concatenated would likely reach 98%+ with zero inference latency.

Add recency weighting: very old priors (10+ years) of the same region are less likely to be relevant than recent ones. This could help with borderline cases.

**Infrastructure:**

Replace in-process dict cache with Redis to survive restarts.
Add structured JSON logging with request IDs for easier debugging of evaluator retries.
Deploy on a persistent host (Railway, Fly.io) rather than a local Cloudflare tunnel.
