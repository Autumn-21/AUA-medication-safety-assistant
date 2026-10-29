"""
Track A - Week 1 data pipeline.

Downloads official FDA drug labels (openFDA) for every drug in drug_list.csv,
normalizes names with RxNorm, splits each label into clinical sections, and
saves clean CSV files ready for EDA and for the RAG index in Week 2.

Usage:
    pip install requests pandas
    python download_data.py                      # uses drug_list.csv
    python download_data.py --drugs my_list.csv  # your own list

Optional: put a free openFDA key in an environment variable OPENFDA_API_KEY
(get one at https://open.fda.gov/apis/authentication/) for higher rate limits.

Outputs (in ./data):
    raw/<drug>.json        the full label as downloaded (for reproducibility)
    drugs.csv              one row per drug: RxCUI, chosen label, DailyMed link
    label_sections.csv     one row per label section: the RAG corpus
    missing_report.csv     drugs that could not be found or normalized
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
RXNORM_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={}"

# openFDA label fields we keep, mapped to friendly section names.
SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",  # older-format labels use this instead
    "precautions": "Precautions",  # older-format labels
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "pediatric_use": "Pediatric Use",
    "geriatric_use": "Geriatric Use",
    "adverse_reactions": "Adverse Reactions",
    "overdosage": "Overdosage",
    "clinical_pharmacology": "Clinical Pharmacology",
    # OTC "Drug Facts" labels (e.g. aspirin) use these instead:
    "do_not_use": "Do Not Use",
    "ask_doctor": "Ask a Doctor Before Use",
    "ask_doctor_or_pharmacist": "Ask a Doctor or Pharmacist Before Use",
    "stop_use": "Stop Use",
    "pregnancy_or_breast_feeding": "Pregnancy or Breast-feeding",
}

# Sections that matter most for flagging; used to score which label to keep.
KEY_SECTIONS = ["contraindications", "drug_interactions", "dosage_and_administration"]

session = requests.Session()


def get_json(url, params, retries=3):
    """GET with simple retry/backoff. Returns parsed JSON or None."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 404:  # openFDA returns 404 for "no matches"
                return None
            if r.status_code == 429:  # rate limited
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"    request failed ({e}); retrying...")
            time.sleep(2 * (attempt + 1))
    return None


# Only real HTML tags are removed. openFDA text is plain text and uses "<" and ">"
# as clinical symbols (e.g. "CrCl <30 mL/min", "QTc >500 ms"), which must be kept.
HTML_TAG = re.compile(r"</?(?:b|i|u|br|p|sup|sub|span|div|table|tbody|thead|tr|td|th|ul|ol|li)\b[^<>]{0,200}>", re.I)


def clean_text(text):
    """Remove real HTML tags only, and collapse whitespace."""
    text = HTML_TAG.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generic_names(label):
    return [g.lower().strip() for g in label.get("openfda", {}).get("generic_name", [])]


def is_combination(label, drug):
    """True if the label is a combination product (e.g. amlodipine AND benazepril).
    Drugs that are themselves combinations (sulfamethoxazole and trimethoprim) are allowed."""
    if " and " in drug or "," in drug:
        return False
    names = generic_names(label)
    # " and ", commas, slashes, or a hyphen joining two names (e.g. "OLMESARTAN MEDOXOMIL-HYDROCHLOROTHIAZIDE")
    return len(names) > 1 or any(re.search(r" and |,|/|[a-z]-[a-z]", g) for g in names)


def label_score(label, drug):
    """Higher score = better label to represent this ingredient."""
    openfda = label.get("openfda", {})
    score = 0
    names = generic_names(label)
    # 1. Single-ingredient product of this drug: most important.
    if any(drug.lower() in g for g in names) and not is_combination(label, drug):
        score += 20
    # 2. Prescription labels have the full clinical sections.
    if "HUMAN PRESCRIPTION DRUG" in openfda.get("product_type", []):
        score += 10
    # 3. Labels that contain the sections we need.
    score += sum(3 for s in KEY_SECTIONS if s in label)
    if "boxed_warning" in label:
        score += 1
    # 4. Newer labels win ties.
    score += int(label.get("effective_time", "0")[:4] or 0) / 10000
    return score


def fetch_label(drug, api_key=None):
    """Search openFDA and return the single best label for this drug.
    Prescription labels are searched first, then all labels, and the results are scored."""
    results = []
    queries = [
        f'openfda.generic_name:"{drug}" AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
        f'openfda.generic_name:"{drug}"',
    ]
    for q in queries:
        params = {"search": q, "limit": 100}
        if api_key:
            params["api_key"] = api_key
        data = get_json(OPENFDA_URL, params)
        results += (data or {}).get("results", [])
    if not results:
        return None
    return max(results, key=lambda lab: label_score(lab, drug))


def fetch_rxcui(drug):
    """Normalize a drug name to its RxNorm concept ID (RxCUI)."""
    data = get_json(RXNORM_URL, {"name": drug, "search": 2})
    ids = (data or {}).get("idGroup", {}).get("rxnormId", [])
    return ids[0] if ids else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drugs", default="drug_list.csv", help="CSV with a 'drug' column")
    parser.add_argument("--out", default="data", help="output folder")
    args = parser.parse_args()

    api_key = os.environ.get("OPENFDA_API_KEY")
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    drug_list = pd.read_csv(args.drugs)
    drug_rows, section_rows, missing = [], [], []

    for i, row in drug_list.iterrows():
        drug = str(row["drug"]).strip().lower()
        category = row.get("category", "")
        print(f"[{i + 1}/{len(drug_list)}] {drug}")

        rxcui = fetch_rxcui(drug)
        label = fetch_label(drug, api_key)
        time.sleep(0.3)  # be polite to the free APIs

        if label is None:
            missing.append({"drug": drug, "problem": "no openFDA label found"})
            continue
        if rxcui is None:
            missing.append({"drug": drug, "problem": "no RxNorm match"})
        if is_combination(label, drug):
            missing.append({"drug": drug, "problem": "only a combination-product label was found; check it"})

        safe_name = re.sub(r"[^a-z0-9]+", "_", drug)
        (out / "raw" / f"{safe_name}.json").write_text(json.dumps(label, indent=2))

        openfda = label.get("openfda", {})
        set_id = label.get("set_id", "")
        url = DAILYMED_URL.format(set_id)
        found = [s for s in SECTIONS if s in label]

        drug_rows.append({
            "drug": drug,
            "category": category,
            "rxcui": rxcui,
            "brand_name": "; ".join(openfda.get("brand_name", [])),
            "manufacturer": "; ".join(openfda.get("manufacturer_name", [])),
            "generic_name": "; ".join(openfda.get("generic_name", [])),
            "product_type": "; ".join(openfda.get("product_type", [])),
            "is_combination_product": is_combination(label, drug),
            "set_id": set_id,
            "label_version": label.get("version", ""),
            "effective_time": label.get("effective_time", ""),
            "dailymed_url": url,
            "n_sections": len(found),
            "has_boxed_warning": "boxed_warning" in label,
            "has_contraindications": "contraindications" in label,
            "has_drug_interactions": "drug_interactions" in label,
            "has_dosage": "dosage_and_administration" in label,
        })

        for field in found:
            text = clean_text(" ".join(label[field]))
            if not text:
                continue
            section_rows.append({
                "section_id": f"{safe_name}::{field}",
                "drug": drug,
                "rxcui": rxcui,
                "section": SECTIONS[field],
                "field": field,
                "text": text,
                "n_chars": len(text),
                "n_words": len(text.split()),
                "set_id": set_id,
                "dailymed_url": url,
            })

    pd.DataFrame(drug_rows).to_csv(out / "drugs.csv", index=False)
    pd.DataFrame(section_rows).to_csv(out / "label_sections.csv", index=False)
    pd.DataFrame(missing, columns=["drug", "problem"]).to_csv(out / "missing_report.csv", index=False)

    print("\nDone.")
    print(f"  drugs with labels : {len(drug_rows)} / {len(drug_list)}")
    print(f"  label sections    : {len(section_rows)}")
    print(f"  problems          : {len(missing)} (see {out / 'missing_report.csv'})")


if __name__ == "__main__":
    main()
