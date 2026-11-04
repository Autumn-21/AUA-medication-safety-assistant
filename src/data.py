"""Loading data and matching drug names.

A patient's medicine list says e.g. "metoprolol succinate | 50 mg | once daily" or "diltiazem ER";
these helpers turn that into the drug names used in our label library ("metoprolol", "diltiazem").
"""
import re
from functools import lru_cache

import pandas as pd

from . import config
from .kidney import crcl_cockcroft_gault, egfr_ckd_epi_2021

# Other names a drug is called by, in patient lists or inside labels.
SYNONYMS = {
    "sulfamethoxazole and trimethoprim": ["sulfamethoxazole and trimethoprim", "sulfamethoxazole/trimethoprim",
                                          "sulfamethoxazole", "trimethoprim", "cotrimoxazole", "co-trimoxazole",
                                          "smx-tmp", "smx/tmp", "bactrim"],
    "potassium chloride": ["potassium chloride", "kcl", "potassium supplement", "potassium supplements",
                           "potassium supplementation"],
    "isosorbide mononitrate": ["isosorbide mononitrate", "isosorbide"],
    "aspirin": ["aspirin", "acetylsalicylic acid"],
    "rifampin": ["rifampin", "rifampicin"],
    "hydrochlorothiazide": ["hydrochlorothiazide", "hctz"],
    "nitroglycerin": ["nitroglycerin", "glyceryl trinitrate"],
    "warfarin": ["warfarin", "coumadin"],
    "apixaban": ["apixaban", "eliquis"],
    "rivaroxaban": ["rivaroxaban", "xarelto"],
    "edoxaban": ["edoxaban", "savaysa"],
    "dronedarone": ["dronedarone", "multaq"],
}

# Words in a patient's medicine name that are not part of the drug name.
_FORM_WORDS = r"\b(er|xl|sr|cr|la|ir|succinate|tartrate|sodium|potassium salt|hydrochloride|hcl|tablets?|" \
              r"capsules?|inhaler|injection|oral|sc|iv)\b"


@lru_cache(maxsize=1)
def drug_names():
    return tuple(pd.read_csv(config.DRUGS).drug.tolist())


def name_patterns():
    """Regex per in-scope drug that matches the drug or any of its synonyms."""
    pats = {}
    for d in drug_names():
        words = SYNONYMS.get(d, [d])
        pats[d] = re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
                             + r")(?![a-z])", re.I)
    return pats


_PATTERNS = None


def patterns():
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = name_patterns()
    return _PATTERNS


@lru_cache(maxsize=1)
def brand_patterns():
    """Common US brand names (data/brand_names.csv) -> generic drug. Used for what users TYPE only,
    not for searching label text, so the benchmark scoring is unchanged."""
    path = config.DATA / "brand_names.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {r.brand.lower(): (re.compile(r"(?<![a-z])" + re.escape(r.brand.lower()) + r"(?![a-z])"), r.drug)
            for r in df.itertuples() if r.drug in drug_names()}


def normalize_drug(name: str):
    """Map a medicine name as written by a user to an in-scope drug, or None if not in our library."""
    text = name.lower().strip()
    for d, pat in patterns().items():
        if pat.search(text):
            return d
    for pat, d in brand_patterns().values():
        if pat.search(text):
            return d
    cleaned = re.sub(_FORM_WORDS, " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned in drug_names() else None


def suggest_drug(name: str):
    """For an unrecognised name, the closest known generic or brand name ("warfrin" -> "warfarin").
    Only a SUGGESTION for the user: it is never used automatically, because a wrong guess
    (e.g. hydroxyzine vs hydralazine) would be unsafe."""
    import difflib
    words = re.sub(_FORM_WORDS, " ", name.lower())
    words = re.findall(r"[a-z][a-z-]{3,}", words)
    known = {d: d for d in drug_names()} | {b: d for b, (_, d) in brand_patterns().items()}
    for w in words:
        match = difflib.get_close_matches(w, list(known), n=1, cutoff=0.8)
        if match:
            return known[match[0]]
    return None


def drugs_in_text(text: str):
    """All in-scope drugs mentioned in a piece of text (e.g. 'warfarin + aspirin 325 mg')."""
    found = {d for d, pat in patterns().items() if pat.search(str(text))}
    # "sulfamethoxazole and trimethoprim" also matches the single words; keep one name
    return found


def parse_medications(med_string: str):
    """'warfarin | 5 mg | once daily; aspirin | 81 mg | once daily' -> list of dicts."""
    meds = []
    for part in str(med_string).split(";"):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split("|")]
        written = bits[0]
        meds.append({
            "written_name": written,
            "drug": normalize_drug(written),
            "dose": bits[1] if len(bits) > 1 else "",
            "frequency": bits[2] if len(bits) > 2 else "",
        })
    return meds


def load_patients(split=None):
    df = pd.read_csv(config.PATIENTS, encoding="utf-8-sig")
    if split:
        df = df[df.split == split]
    return df


def patient_record(row) -> dict:
    """Everything the systems need about one patient, with kidney function calculated in code."""
    meds = parse_medications(row.medications)
    return {
        "patient_id": row.patient_id,
        "age": int(row.age),
        "sex": row.sex,
        "weight_kg": float(row.weight_kg),
        "serum_creatinine_mg_dl": float(row.serum_creatinine_mg_dl),
        "crcl_ml_min": round(crcl_cockcroft_gault(row.age, row.weight_kg, row.serum_creatinine_mg_dl, row.sex), 1),
        "egfr": round(egfr_ckd_epi_2021(row.age, row.serum_creatinine_mg_dl, row.sex), 1),
        "pregnant": str(row.pregnant),
        "conditions": str(row.conditions),
        "allergies": str(row.allergies),
        "medications": meds,
        "in_scope_drugs": sorted({m["drug"] for m in meds if m["drug"]}),
        "out_of_scope": [m["written_name"] for m in meds if not m["drug"]],
    }


def load_expected(split=None):
    df = pd.read_csv(config.EXPECTED_FLAGS, encoding="utf-8-sig")
    if split:
        df = df[df.split == split]
    df = df.copy()
    df["drug_set"] = df.drugs.map(lambda s: frozenset(drugs_in_text(s)))
    return df


def load_acceptable_extras():
    if not config.ACCEPTABLE_EXTRAS.exists():
        return {}
    df = pd.read_csv(config.ACCEPTABLE_EXTRAS, encoding="utf-8-sig")
    out = {}
    for r in df.itertuples():
        out.setdefault(r.patient_id, []).append(frozenset(drugs_in_text(r.drugs)))
    return out


@lru_cache(maxsize=1)
def drug_classes():
    df = pd.read_csv(config.DRUG_CLASSES)
    return {r.drug: [c.strip() for c in r.classes.split(";")] for r in df.itertuples()}


def patient_summary(p: dict) -> str:
    """Readable patient description used in AI prompts."""
    med_lines = "\n".join(
        f"- {m['written_name']} {m['dose']} {m['frequency']}".rstrip()
        + ("" if m["drug"] else "  [NOT IN OUR LABEL LIBRARY]") for m in p["medications"])
    return (f"Age: {p['age']}  Sex: {p['sex']}  Weight: {p['weight_kg']} kg\n"
            f"Serum creatinine: {p['serum_creatinine_mg_dl']} mg/dL  ->  CrCl (Cockcroft-Gault): {p['crcl_ml_min']} mL/min, "
            f"eGFR (CKD-EPI 2021): {p['egfr']} mL/min/1.73m2  (calculated by code)\n"
            f"Pregnant: {p['pregnant']}\nConditions: {p['conditions']}\nAllergies: {p['allergies']}\n"
            f"Medications:\n{med_lines}")
