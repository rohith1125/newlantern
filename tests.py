"""
Regression tests for the highest-risk logic in predictor.py.
Run with: python tests.py
"""

import sys
sys.path.insert(0, ".")

from predictor import parse_study, predict_relevance


def check(description, current, prior, expected):
    cur = parse_study(current)
    pri = parse_study(prior)
    result = predict_relevance(cur, pri)
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        print(f"  {status}  [{description}]")
        print(f"    CUR:  {current}")
        print(f"    PRI:  {prior}")
        print(f"    expected={expected}  got={result}")
    return result == expected


def run():
    failures = 0
    cases = []

    def t(desc, cur, pri, exp):
        cases.append((desc, cur, pri, exp))

    # --- same region, same modality ---
    t("brain/brain same",        "MRI BRAIN WITHOUT CONTRAST",               "MRI BRAIN WITHOUT CONTRAST",          True)
    t("chest/chest same",        "CT CHEST WITH CONTRAST",                   "CT CHEST WITHOUT CONTRAST",           True)
    t("abdomen/abdomen same",    "CT ABDOMEN WITH CONTRAST",                 "CT ABDOMEN WITHOUT CONTRAST",         True)

    # --- same region, different modality ---
    t("brain CT vs MRI",         "CT HEAD WITHOUT CNTRST",                   "MRI BRAIN WITHOUT CONTRAST",          True)
    t("chest CT vs XR",          "CT CHEST WITH CONTRAST",                   "XR Chest 1V Frontal Only",            True)

    # --- clearly different regions ---
    t("brain vs chest",          "MRI BRAIN WITHOUT CONTRAST",               "CT CHEST WITHOUT CONTRAST",           False)
    t("brain vs knee",           "MRI BRAIN WITHOUT CONTRAST",               "MRI KNEE WITHOUT CONTRAST",           False)
    t("chest vs abdomen",        "CT CHEST WITHOUT CONTRAST",                "CT ABDOMEN WITHOUT CONTRAST",         False)

    # --- spine segment independence ---
    t("lumbar vs cervical",      "MRI LUMBAR SPINE WITHOUT CONTRAST",        "MRI CERVICAL SPINE WITHOUT CONTRAST", False)
    t("lumbar vs thoracic",      "MRI LUMBAR SPINE WITHOUT CONTRAST",        "MRI THORACIC SPINE WITHOUT CONTRAST", False)
    t("whole spine vs lumbar",   "MRI WHOLE SPINE",                          "MRI LUMBAR SPINE",                    True)
    t("lumbar vs whole spine",   "MRI LUMBAR SPINE",                         "MRI WHOLE SPINE",                     True)
    t("lumbar vs sacral",        "MRI LUMBAR SPINE",                         "MRI SACRUM",                          True)

    # --- breast laterality ---
    t("breast RT vs RT",         "MAM diagnostic RT with tomo",              "MAM diagnostic RT with tomo",         True)
    t("breast LT vs LT",         "MAM diagnostic LT with tomo",              "MAM diagnostic LT with tomo",         True)
    t("breast RT vs LT",         "MAM diagnostic RT with tomo",              "MAM diagnostic LT with tomo",         False)
    t("breast LT vs RT",         "MAM diagnostic LT with tomo",              "MAM diagnostic RT with tomo",         False)
    t("breast BI vs RT",         "MAM screen BI with tomo",                  "MAM diagnostic RT with tomo",         True)
    t("breast RT vs BI",         "MAM diagnostic RT with tomo",              "MAM screen BI with tomo",             True)
    t("breast unknown vs RT",    "MAM SCREEN 3D",                            "MAM diagnostic RT with tomo",         True)

    # --- breast vs chest isolation ---
    t("breast vs chest CT",      "MAM diagnostic RT with tomo",              "CT CHEST WITH CONTRAST",              False)
    t("breast vs chest XR",      "MAM diagnostic RT with tomo",              "CHEST 2 VIEW FRONTAL & LATRL",        False)

    # --- abdomen/pelvis separation ---
    t("abdomen vs pelvis",       "CT ABDOMEN WITHOUT CONTRAST",              "MRI PELVIS WITHOUT CONTRAST",         False)
    t("pelvis vs abdomen",       "MRI PELVIS WITHOUT CONTRAST",              "CT ABDOMEN WITHOUT CONTRAST",         False)
    t("abd_pelvis vs abdomen",   "CT ABD/PELVIS WITHOUT CONTRAST",           "CT ABDOMEN WITHOUT CONTRAST",         True)
    t("abd_pelvis vs pelvis",    "CT ABD/PELVIS WITHOUT CONTRAST",           "MRI PELVIS WITHOUT CONTRAST",         True)

    # --- DXA isolation ---
    t("DXA vs DXA",              "DXA (Hip/Spine Only)",                     "BONE DENSITY (HIP/SPINE)",            True)
    t("DXA vs abdomen",          "DXA (Hip/Spine Only)",                     "XR Abdomen 1V",                       False)
    t("abdomen vs DXA",          "CT ABDOMEN WITH CONTRAST",                 "DXA (Hip/Spine Only)",                False)

    # --- cardiac/chest ---
    t("cardiac vs cardiac",      "ECHO 2D Mmode transthorac TTE",            "ECHO 2D Mmode transthorac TTE",       True)
    t("chest vs chest CT",       "CT CHEST WITH CONTRAST",                   "CT CHEST WITHOUT CONTRAST",           True)
    t("cardiac vs chest",        "ECHO 2D Mmode transthorac TTE",            "CT CHEST WITH CONTRAST",              False)
    t("chest vs cardiac",        "CT CHEST WITH CONTRAST",                   "ECHO 2D Mmode transthorac TTE",       False)

    # --- adjacent extremity ---
    t("knee vs femur",           "MRI KNEE WITHOUT CONTRAST",                "XR FEMUR",                            True)
    t("wrist vs forearm",        "XR WRIST",                                 "XR FOREARM",                          True)
    t("shoulder vs knee",        "MRI SHOULDER",                             "MRI KNEE",                            False)

    # --- whole-body / bone scan ---
    t("bone scan vs chest CT",   "Bone Scan",                                "CT CHEST WITH CONTRAST",              True)
    t("bone scan vs knee",       "Bone Scan",                                "MRI KNEE WITHOUT CONTRAST",           True)
    t("bone scan vs breast",     "Bone Scan",                                "MAM screen BI with tomo",             False)

    # --- abbreviated/variant descriptions ---
    t("CERV spine",              "MRI CERV SPINE WITHOUT CNTRST",            "CT CERVICAL SPINE WITHOUT CONTRAST",  True)
    t("ribs vs chest",           "XR ribs RT",                               "CT CHEST WITHOUT CONTRAST",           True)
    t("thoracentesis vs chest",  "THORACENTESIS (CT GUIDED)",                "CT CHEST WITHOUT CONTRAST",           True)

    # --- unknown region defaults to false ---
    t("unknown vs known",        "CT guided FNA",                            "CT CHEST WITHOUT CONTRAST",           False)
    t("unknown vs unknown",      "PROCEDURE NOTE",                           "CONSULT REQUEST",                     False)

    for args in cases:
        if not check(*args):
            failures += 1

    total = len(cases)
    passed = total - failures
    print(f"\n{passed}/{total} passed", "- ALL PASS" if failures == 0 else f"- {failures} FAILED")
    return failures == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
