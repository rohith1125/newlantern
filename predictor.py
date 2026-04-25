import re
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Optional


MODALITY_PATTERNS = {
    "CT":    re.compile(r"\bCT\b|CAT SCAN|\bCAT\b"),
    "MRI":   re.compile(r"\bMRI\b|\bMR\b"),
    "XR":    re.compile(r"\bX[-\s]?RAY\b|\bXR\b|\bCXR\b|\bCHEST PA\b|\bCHEST\s+\d|\bCHEST AP\b"),
    "US":    re.compile(r"\bULTRAS\w*\b|\bUS\b(?!\s*\w{4,})|\bDOPPLER\b|\bVAS\b|\bDUPLEX\b"),
    "ECHO":  re.compile(r"\bECHO\b|\bTTE\b|\bTEE\b"),
    "NM":    re.compile(r"\bNUCLEAR\b|\bNM\b|\bSCINTI\w*\b|\bSPECT\b|\bBONE SCAN\b"),
    "PET":   re.compile(r"\bPET\b"),
    "MAMMO": re.compile(r"\bMAMMO\w*\b|\bMAM\b"),
    "DXA":   re.compile(r"\bDXA\b|\bBONE DENSITY\b|\bDEXA\b"),
    "FLUORO":re.compile(r"\bFLUORO\w*\b|\bBAR\w+ SWALLOW\b|\bUGI\b"),
    "ANGIO": re.compile(r"\bANGIO\w*\b|\bARTERIO\w*\b"),
}

REGION_PATTERNS = {
    # brain / neurology — does NOT include carotid/vascular (separate neurovascular region)
    "brain":          re.compile(r"\bBRAIN\b|\bCRANI\w*\b|\bINTRACRANI\w*\b|\bSTROKE\b|\bEEG\b|\bCEREBR\w*\b|\bSEIZURE\b"),
    "head":           re.compile(r"\bHEAD\b"),
    # neurovascular — angio/doppler/carotid — matches each other but NOT plain brain parenchyma
    "neurovascular":  re.compile(r"\bCAROTID\b|\bTRANSCRANIAL\b|\bTCD\b|\bANGIO\b.*\b(?:HEAD|NECK|BRAIN|CEREB)\b|\b(?:HEAD|NECK|BRAIN).*\bANGIO\b|\bMRA\b|\bCTA\s+HEAD\b|\bCTA\s+NECK\b"),
    "face":           re.compile(r"\bFACE\b|\bFACIAL\b|\bMAXILLO\w*\b|\bMANDIBLE\b"),
    "orbit":          re.compile(r"\bORBIT\w*\b|\bEYE\b"),
    "sinus":          re.compile(r"\bSINUS\w*\b|\bSINO\w*\b"),
    "neck":           re.compile(r"\bNECK\b|\bSOFT TISSUE NECK\b|\bTHYROID\b|\bSALIVARY\b|\bPAROTID\b"),
    "chest":          re.compile(r"\bCHEST\b|\bTHORAX\b(?!\s*SPINE|\s*SPI)|\bLUNG\w*\b|\bPLEUR\w*\b|\bMEDIASTIN\w*\b|\bPULMON\w*\b|\bPUL\b|\bESOPHAG\w*\b|\bESOP\b|\bRIBS?\b|\bSTERNUM\b|\bTHORACENTESIS\b"),
    "cardiac":        re.compile(r"\bCARDIAC\b|\bHEART\b|\bCARDIO\w*\b|\bCORONARY\b|\bMYO\w*\b|\bECHO\b|\bTTE\b|\bTEE\b|\bCALC SCREEN\b|\bCALCIUM SCORE\b|\bFFR\b|\bNMMYO\b|\bNM\s*MYO\b|\bMYOCARDIAL\b"),
    "breast":         re.compile(r"\bBREAST\b|\bMAMMO\w*\b|\bMAM\b|\bBILAT\s+SCREEN\b|\bSCREEN\s+COMP\b|\bCOMBOHD\b|\bDIGITAL\s+SCREEN\w*\b|\bSCREENER\b|\bSTANDARD\s+SCREEN\w*\b|\bSEED\s+LOCAL\w*\b|\bDIAG\s+TARGET\b"),
    "abdomen":        re.compile(r"\bABDOMEN\b|\bABD\b(?!\s*PELVIC|\s*PEL)|\bABDOMINAL\b|\bENTEROGRAPH\w*\b|\bPARACENTES\w*\b|\bPERITONEAL\b|\bRETROPERITONEAL\b|\bDRAINAGE.*COLLECTION\b|\bBIOPSY.*ABD\w*\b"),
    "pelvis":         re.compile(r"\bPELV\w*\b|\bHIP\w*\b|\bENDOVAGINAL\b|\bTRANSVAGINAL\b|\bENDOCAV\w*\b|\bUTER\w*\b|\bOVAR\w*\b|\bBLADDER\b"),
    "abdomen_pelvis": re.compile(r"\bABD\w*[\s_/&]+PEL\w*\b|\bA/P\b|\bABD_PEL\w*\b|\bABD/PEL\w*\b|\bABDOMEN.{0,5}PELVIS\b"),
    "spine_cervical": re.compile(r"\bCERVICAL\b(?!\s*SOFT)|\bCERVICL\b|\bCERV\b|\bC[-\s]?SPINE\b|\bCSPINE\b"),
    "spine_thoracic": re.compile(r"\bTHORACIC\s*SPINE\b|\bT[-\s]?SPINE\b|\bTSPINE\b"),
    "spine_lumbar":   re.compile(r"\bLUMBAR\b|\bL[-\s]?SPINE\b|\bLSPINE\b|\bLS SPINE\b|\bSCOLIOS\w*\b"),
    "spine_sacral":   re.compile(r"\bSACR\w*\b|\bCOCCYX\b|\bSI JOINT\b"),
    "whole_spine":    re.compile(r"\bWHOLE SPINE\b|\bENTIRE SPINE\b|\bFULL SPINE\b|\bSPINE SURVEY\b"),
    "whole_body":     re.compile(r"\bSKULL TO THIGH\b|\bSKULLBASE\b|\bWHOLE BODY\b|\bFULL BODY\b|\bSKULL TO (?:MID)?THIGH\b|\bBONE SCAN\b|\bNM BONE\b|\bPETCT_SKULL\w*\b"),
    "shoulder":       re.compile(r"\bSHOULDER\b|\bROTATOR\b|\bACROMI\w*\b|\bCLAVICLE\b|\bAC JOINT\b"),
    "elbow":          re.compile(r"\bELBOW\b|\bOLECRANON\b"),
    "wrist":          re.compile(r"\bWRIST\b|\bCARPAL\b"),
    "hand":           re.compile(r"\bHAND\b|\bFINGER\b|\bTHUMB\b"),
    "forearm":        re.compile(r"\bFOREARM\b|\bRADIUS\b|\bULNA\b"),
    "humerus":        re.compile(r"\bHUMERUS\b|\bUPPER ARM\b|\bUPPR\b"),
    "upper_extremity":re.compile(r"\bUPPER\s+EXTREM\w*\b|\bUE\b(?!\s*\w{5,})|\bUPPR\s+(?:RT|LT|RIGHT|LEFT)\s+EXTREM\w*\b"),
    "knee":           re.compile(r"\bKNEE\b|\bPATELL\w*\b|\bMENISC\w*\b"),
    "ankle":          re.compile(r"\bANKLE\b|\bACHILLES\b"),
    "foot":           re.compile(r"\bFOOT\b|\bFEET\b|\b(?:BIG\s)?TOE\b|\bCALCAN\w*\b|\bMETATARSAL\b"),
    "lower_leg":      re.compile(r"\bTIBIA\b|\bFIBULA\b|\bLEG\b(?!\s*BI)"),
    "femur":          re.compile(r"\bFEMUR\b|\bFEMORAL\b|\bTHIGH\b"),
    "lower_extremity":re.compile(r"\bLOWER\s+EXTREM\w*\b|\bLE\b(?!\s*\w{5,})|\bCT\s+LE\b"),
    "liver":          re.compile(r"\bLIVER\b|\bHEPAT\w*\b|\bBILIARY\b|\bGALLBLADDER\b|\bMRCP\b|\bCHOLANGIO\w*\b"),
    "kidney":         re.compile(r"\bKIDNEY\b|\bRENAL\b|\bNEPHR\w*\b|\bUS\s+KIDNEY\w*\b"),
    "venous_leg":     re.compile(r"\bVENOUS\b.*\b(?:LE|LEG|LOWER|BI)\b|\bDVT\b|\bVAS.*(?:LE|LEG)\b|\bLEG.*VENOUS\b|\bVEINS.*LE\b"),
    "upper_gi":       re.compile(r"\bBARIUM\b|\bSWALLOW\b|\bUPPER GI\b|\bUGI\b|\bESOPHAGR\w*\b"),
    "dxa":            re.compile(r"\bDXA\b|\bBONE DENSITY\b|\bDEXA\b"),
}

REGION_GROUPS: dict[str, list[str]] = {
    "brain":          ["brain", "head"],
    "head":           ["brain", "head"],
    "neurovascular":  ["neurovascular"],   # angio/carotid/TCD — matches only other neurovascular
    "neck":           ["neck", "spine_cervical"],
    "spine_cervical": ["spine_cervical", "neck"],
    "spine_thoracic": ["spine_thoracic"],        # doesn't match lumbar through whole_spine
    "spine_lumbar":   ["spine_lumbar", "spine_sacral"],
    "spine_sacral":   ["spine_sacral", "spine_lumbar"],
    "whole_spine":    ["whole_spine", "spine_cervical", "spine_thoracic", "spine_lumbar", "spine_sacral"],
    "chest":          ["chest"],
    "cardiac":        ["cardiac"],
    # abdomen and pelvis do NOT directly cross-match — only via explicit abdomen_pelvis study
    "abdomen":        ["abdomen", "liver", "kidney"],
    "pelvis":         ["pelvis"],
    "abdomen_pelvis": ["abdomen_pelvis", "abdomen", "pelvis", "liver", "kidney"],
    "liver":          ["liver", "abdomen"],
    "kidney":         ["kidney", "abdomen"],
    "breast":         ["breast"],
    "upper_gi":       ["upper_gi"],
    "upper_extremity": ["upper_extremity", "shoulder", "humerus", "elbow", "forearm", "wrist", "hand"],
    "lower_extremity": ["lower_extremity", "femur", "knee", "lower_leg", "ankle", "foot"],
    # upper extremity adjacency
    "shoulder":       ["shoulder", "humerus", "upper_extremity"],
    "humerus":        ["humerus", "shoulder", "elbow", "forearm", "upper_extremity"],
    "elbow":          ["elbow", "humerus", "forearm"],
    "forearm":        ["forearm", "elbow", "wrist", "humerus"],
    "wrist":          ["wrist", "forearm", "hand"],
    "hand":           ["hand", "wrist"],
    # lower extremity adjacency
    "femur":          ["femur", "pelvis", "knee", "lower_extremity"],
    "knee":           ["knee", "femur", "lower_leg", "lower_extremity"],
    "lower_leg":      ["lower_leg", "knee", "ankle", "lower_extremity"],
    "ankle":          ["ankle", "lower_leg", "foot", "lower_extremity"],
    "foot":           ["foot", "ankle", "lower_extremity"],
    # bone scan / whole-body PET
    "whole_body":     [
        "whole_body", "brain", "head", "chest",
        "abdomen", "pelvis", "abdomen_pelvis", "liver", "kidney",
        "spine_cervical", "spine_thoracic", "spine_lumbar", "spine_sacral", "whole_spine",
        "shoulder", "humerus", "elbow", "forearm", "wrist", "hand",
        "femur", "knee", "lower_leg", "ankle", "foot",
        "neck", "neurovascular",
    ],
    "dxa":            ["dxa"],
    "venous_leg":     ["venous_leg", "lower_leg", "ankle", "foot"],
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
    current_expanded: set[str] = set()
    for r in current_regions:
        current_expanded.update(REGION_GROUPS.get(r, [r]))

    prior_expanded: set[str] = set()
    for r in prior_regions:
        prior_expanded.update(REGION_GROUPS.get(r, [r]))

    return bool(current_expanded & prior_expanded)


_HEAD_AND_NECK_RE = re.compile(r"\bHEAD\s+AND\s+NECK\b|\bHEAD\s*[&/]\s*NECK\b", re.IGNORECASE)
_LAT_RT = re.compile(r"\bRT\b|\bRIGHT\b", re.IGNORECASE)
_LAT_LT = re.compile(r"\bLT\b|\bLEFT\b", re.IGNORECASE)
_LAT_BI = re.compile(r"\bBI\b|\bBILAT\w*\b|\bBOTH\b", re.IGNORECASE)


def _breast_laterality(desc: str) -> str:
    if _LAT_BI.search(desc):
        return "BI"
    if _LAT_RT.search(desc):
        return "RT"
    if _LAT_LT.search(desc):
        return "LT"
    return "UNKNOWN"


def predict_relevance(current: StudyFeatures, prior: StudyFeatures) -> bool:
    if not current.regions or not prior.regions:
        return False

    # DXA only matches other DXA
    if "dxa" in current.regions or "dxa" in prior.regions:
        return "dxa" in current.regions and "dxa" in prior.regions

    # "head and neck" soft tissue studies (US/CT) are neck studies, not intracranial.
    # If the description explicitly says "head and neck" and the modality is soft tissue,
    # strip "head" from the effective regions to avoid false matches with brain/head CT.
    def effective_regions(feat: StudyFeatures) -> list[str]:
        if _HEAD_AND_NECK_RE.search(feat.description) and feat.modality in ("US", None):
            return [r for r in feat.regions if r != "head"]
        return feat.regions

    cur_regions = effective_regions(current)
    pri_regions = effective_regions(prior)

    if not cur_regions or not pri_regions:
        return False

    # breast laterality: RT and LT are not relevant to each other
    cur_breast = "breast" in cur_regions
    pri_breast = "breast" in pri_regions
    if cur_breast and pri_breast:
        cur_lat = _breast_laterality(current.description)
        pri_lat = _breast_laterality(prior.description)
        if cur_lat != "UNKNOWN" and pri_lat != "UNKNOWN":
            if cur_lat != "BI" and pri_lat != "BI" and cur_lat != pri_lat:
                return False

    return _regions_overlap(cur_regions, pri_regions)
