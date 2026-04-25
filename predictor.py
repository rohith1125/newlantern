import re
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Optional


MODALITY_PATTERNS = {
    "CT": re.compile(r"\bCT\b|CAT SCAN|\bCAT\b"),
    "MRI": re.compile(r"\bMRI\b|\bMR\b"),
    "XR": re.compile(r"\bX[-\s]?RAY\b|\bXR\b|\bCXR\b|\bCHEST PA\b|\bPA &\b|\bAP &\b"),
    "US": re.compile(r"\bULTRAS\w*\b|\bUS\b(?! -)|\bDOPPLER\b|\bECHO\b"),
    "NM": re.compile(r"\bNUCLEAR\b|\bNM\b|\bSCINTI\w*\b|\bSPECT\b|\bBONE SCAN\b"),
    "PET": re.compile(r"\bPET\b"),
    "MAMMO": re.compile(r"\bMAMMO\w*\b"),
    "FLUORO": re.compile(r"\bFLUORO\w*\b|\bBAR\w+ SWALLOW\b|\bUGI\b"),
    "ANGIO": re.compile(r"\bANGIO\w*\b|\bARTERIO\w*\b"),
}

REGION_PATTERNS = {
    "brain": re.compile(r"\bBRAIN\b|\bCRANI\w*\b|\bINTRACRANI\w*\b"),
    "head": re.compile(r"\bHEAD\b"),
    "face": re.compile(r"\bFACE\b|\bFACIAL\b|\bMAXILLO\w*\b|\bMANDIBLE\b"),
    "orbit": re.compile(r"\bORBIT\w*\b|\bEYE\b"),
    "sinus": re.compile(r"\bSINUS\w*\b|\bSINO\w*\b"),
    "neck": re.compile(r"\bNECK\b|\bCERVICAL SOFT TISSUE\b|\bTHYROID\b|\bTHYROID\b|\bSALIVARY\b"),
    "chest": re.compile(r"\bCHEST\b|\bTHORAX\b|\bTHORACIC\b|\bLUNG\w*\b|\bPLEUR\w*\b|\bMEDIASTIN\w*\b"),
    "cardiac": re.compile(r"\bCARDIAC\b|\bHEART\b|\bCARDIO\w*\b|\bCORONARY\b|\bMYO\w*\b"),
    "abdomen": re.compile(r"\bABDOMEN\b|\bABD\b|\bABDOMINAL\b"),
    "pelvis": re.compile(r"\bPELV\w*\b|\bHIP\w*\b"),
    "spine_cervical": re.compile(r"\bCERVICAL\b.*\bSPINE\b|\bC[-\s]?SPINE\b|\bC[-\s]?\d"),
    "spine_thoracic": re.compile(r"\bTHORACIC\b.*\bSPINE\b|\bT[-\s]?SPINE\b"),
    "spine_lumbar": re.compile(r"\bLUMBAR\b|\bL[-\s]?SPINE\b|\bL[-\s]?\d"),
    "spine_sacral": re.compile(r"\bSACR\w*\b|\bCOCCYX\b|\bSI JOINT\b"),
    "whole_spine": re.compile(r"\bWHOLE SPINE\b|\bENTIRE SPINE\b|\bFULL SPINE\b"),
    "shoulder": re.compile(r"\bSHOULDER\b|\bROTATOR\b|\bACROMI\w*\b|\bCLAVICLE\b"),
    "elbow": re.compile(r"\bELBOW\b|\bOLECRANON\b"),
    "wrist": re.compile(r"\bWRIST\b|\bCARPAL\b"),
    "hand": re.compile(r"\bHAND\b|\bFINGER\b|\bTHUMB\b"),
    "forearm": re.compile(r"\bFOREARM\b|\bRADIUS\b|\bULNA\b"),
    "humerus": re.compile(r"\bHUMERUS\b|\bUPPER ARM\b"),
    "upper_extremity": re.compile(r"\bUPPER EXTREM\w*\b"),
    "knee": re.compile(r"\bKNEE\b|\bPATELL\w*\b|\bMENISC\w*\b"),
    "ankle": re.compile(r"\bANKLE\b|\bACHILLES\b"),
    "foot": re.compile(r"\bFOOT\b|\bFEET\b|\bTOE\b|\bCALCAN\w*\b|\bMETATARSAL\b"),
    "tibia": re.compile(r"\bTIBIA\b|\bFIBULA\b|\bLEG\b(?!S)"),
    "femur": re.compile(r"\bFEMUR\b|\bFEMORAL\b|\bTHIGH\b"),
    "lower_extremity": re.compile(r"\bLOWER EXTREM\w*\b"),
    "breast": re.compile(r"\bBREAST\b"),
    "liver": re.compile(r"\bLIVER\b|\bHEPAT\w*\b|\bBILIARY\b"),
    "kidney": re.compile(r"\bKIDNEY\b|\bRENAL\b|\bNEPHR\w*\b"),
    "abdomen_pelvis": re.compile(r"\bABD\w*\s*(?:AND|&|/)\s*PELV\w*\b|\bA/P\b"),
}

REGION_GROUPS = {
    "brain": ["brain", "head"],
    "head": ["brain", "head"],
    "spine_cervical": ["spine_cervical"],
    "spine_thoracic": ["spine_thoracic"],
    "spine_lumbar": ["spine_lumbar", "spine_sacral"],
    "spine_sacral": ["spine_sacral", "spine_lumbar"],
    "whole_spine": ["whole_spine", "spine_cervical", "spine_thoracic", "spine_lumbar", "spine_sacral"],
    "abdomen": ["abdomen", "abdomen_pelvis", "liver", "kidney"],
    "pelvis": ["pelvis", "abdomen_pelvis", "spine_sacral"],
    "abdomen_pelvis": ["abdomen_pelvis", "abdomen", "pelvis", "liver", "kidney"],
    "liver": ["liver", "abdomen", "abdomen_pelvis"],
    "kidney": ["kidney", "abdomen", "abdomen_pelvis"],
    "cardiac": ["cardiac", "chest"],
    "chest": ["chest", "cardiac"],
    "upper_extremity": ["upper_extremity", "shoulder", "elbow", "wrist", "hand", "humerus", "forearm"],
    "lower_extremity": ["lower_extremity", "knee", "ankle", "foot", "tibia", "femur"],
}


@dataclass
class StudyFeatures:
    description: str
    modality: Optional[str]
    regions: list[str] = field(default_factory=list)
    study_date: Optional[date] = None


def extract_modality(description: str) -> Optional[str]:
    upper = description.upper()
    for modality, pattern in MODALITY_PATTERNS.items():
        if pattern.search(upper):
            return modality
    return None


def extract_regions(description: str) -> list[str]:
    upper = description.upper()
    found = []
    for region, pattern in REGION_PATTERNS.items():
        if pattern.search(upper):
            found.append(region)
    return found


def parse_study(description: str, study_date: Optional[str] = None) -> StudyFeatures:
    modality = extract_modality(description)
    regions = extract_regions(description)

    parsed_date = None
    if study_date:
        try:
            parsed_date = datetime.fromisoformat(study_date).date()
        except ValueError:
            pass

    return StudyFeatures(
        description=description,
        modality=modality,
        regions=regions,
        study_date=parsed_date,
    )


def _regions_overlap(current_regions: list[str], prior_regions: list[str]) -> bool:
    if not current_regions or not prior_regions:
        return True  # can't tell → assume relevant

    current_expanded = set(current_regions)
    for r in current_regions:
        current_expanded.update(REGION_GROUPS.get(r, [r]))

    prior_expanded = set(prior_regions)
    for r in prior_regions:
        prior_expanded.update(REGION_GROUPS.get(r, [r]))

    return bool(current_expanded & prior_expanded)


def predict_relevance(current: StudyFeatures, prior: StudyFeatures) -> bool:
    # no region info on either side → can't reject
    if not current.regions and not prior.regions:
        return True

    # regions don't overlap → not relevant
    if not _regions_overlap(current.regions, prior.regions):
        return False

    # same region — almost always relevant regardless of modality
    return True
