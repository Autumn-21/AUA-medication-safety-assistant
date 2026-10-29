"""Checks that every expected flag in the test set is backed by real label text.

For each row in expected_flags.csv it looks up the label section
(source_drug + source_section) in data/label_sections.csv and checks that the
evidence_phrase appears in it. Run it again every time you re-download labels.

Usage:
    python verify_test_set.py                         # default paths
    python verify_test_set.py --labels data/label_sections.csv --flags test_set/expected_flags.csv
"""
import argparse
import re
import sys

import pandas as pd


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/label_sections.csv")
    ap.add_argument("--flags", default="test_set/expected_flags.csv")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    flags = pd.read_csv(args.flags, encoding="utf-8-sig")
    texts = {(r.drug, r.field): norm(r.text) for r in labels.itertuples()}
    by_drug = {}
    for (drug, field), t in texts.items():
        by_drug.setdefault(drug, []).append((field, t))

    failed = 0
    for f in flags.itertuples():
        phrase = norm(f.evidence_phrase)
        text = texts.get((f.source_drug, f.source_section))
        if text is not None and phrase in text:
            continue
        failed += 1
        where = [field for field, t in by_drug.get(f.source_drug, []) if phrase in t]
        hint = f"found instead in: {', '.join(where)}" if where else "not found in any section of this label"
        print(f"FAIL {f.flag_id} {f.patient_id} [{f.source_drug} / {f.source_section}] "
              f"'{f.evidence_phrase}' -> {hint}")

    print(f"\n{len(flags) - failed}/{len(flags)} expected flags are backed by the label text.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
