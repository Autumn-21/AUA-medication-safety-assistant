"""The working pipeline for one patient: medicines in -> warnings with sources out.

Run:
  python -m src.run --patient P001                    # a practice patient (RAG system)
  python -m src.run --patient P001 --system keyword   # compare with a baseline
  python -m src.run --meds "warfarin | 5 mg | daily; naproxen | 500 mg | twice daily" --age 73 --sex F \\
                    --weight 64 --scr 0.9 --conditions "atrial fibrillation"
"""
import argparse
import sys
from types import SimpleNamespace

from . import config, systems
from .data import load_patients, patient_record


def print_result(out):
    print(f"\nPatient {out['patient_id']}  ({out['meta']['system']})")
    if not out["flags"]:
        print("  No warnings found in the label evidence.")
    for i, f in enumerate(out["flags"], 1):
        print(f"\n  {i}. [{f['severity'].upper()}] {f['flag_type']}: {' + '.join(f['drugs'])}")
        print(f"     {f.get('explanation', '')}")
        if f.get("recommendation"):
            print(f"     Recommendation: {f['recommendation']}")
        for c in f["citations"]:
            mark = "verified" if c.get("verified") else "NOT VERIFIED"
            section = config.SECTION_NAMES.get(c.get("field"), c.get("field") or c.get("claimed_section"))
            print(f"     Source ({mark}): {c.get('drug')} label, {section}: \"{c.get('quote', '')[:160]}\"")
            if c.get("dailymed_url"):
                print(f"       {c['dailymed_url']}")
    if out.get("removed_flags"):
        print(f"\n  {len(out['removed_flags'])} warning(s) removed: their quotes were not found in the cited label text.")
    if out.get("not_in_library"):
        print(f"\n  Not in our label library (not checked): {', '.join(out['not_in_library'])}")
    print("\n  For pharmacist decision support only; not for clinical use.")


def main():
    if hasattr(sys.stdout, "reconfigure"):          # label text contains symbols like "≤"; never crash printing them
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Check one patient's medicines")
    ap.add_argument("--patient", help="practice patient id, e.g. P001")
    ap.add_argument("--system", default="rag_v3", choices=["rag_v3", "rag_v2", "rag", "keyword", "llm_only"],
                    help="rag_v3 = final version (default); rag_v2 = Week 3; rag = Week 2")
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    ap.add_argument("--meds"), ap.add_argument("--age", type=float), ap.add_argument("--sex", default="F")
    ap.add_argument("--weight", type=float), ap.add_argument("--scr", type=float, help="serum creatinine mg/dL")
    ap.add_argument("--conditions", default="none"), ap.add_argument("--allergies", default="none")
    ap.add_argument("--pregnant", default="No")
    args = ap.parse_args()

    if args.patient:
        rows = load_patients()
        row = rows[rows.patient_id == args.patient]
        if row.empty:
            raise SystemExit(f"Unknown patient {args.patient}")
        patient = patient_record(row.iloc[0])
    elif args.meds:
        if None in (args.age, args.weight, args.scr):
            raise SystemExit("--age, --weight and --scr are needed to calculate kidney function")
        patient = patient_record(SimpleNamespace(
            patient_id="NEW", age=args.age, sex=args.sex, weight_kg=args.weight, serum_creatinine_mg_dl=args.scr,
            pregnant=args.pregnant, conditions=args.conditions, allergies=args.allergies, medications=args.meds))
    else:
        raise SystemExit("Give --patient P001, or --meds with patient details")

    if args.system == "keyword":
        out = systems.keyword_system(patient)
    elif args.system == "llm_only":
        out = systems.llm_only_system(patient)
    else:
        from .retrieval import Retriever
        variant = {"rag": "v1", "rag_v2": "v2", "rag_v3": "v3"}[args.system]
        out = systems.rag_system(patient, Retriever(args.mode), variant=variant)
    print_result(out)


if __name__ == "__main__":
    main()
