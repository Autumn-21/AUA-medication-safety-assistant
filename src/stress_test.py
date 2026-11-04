"""Week 3 - Edge-case and bias testing ("stress test") and the error analysis log.

Part A (no AI): can the system recognise brand names and refuse to guess misspellings?
Part B: runs the system on the tricky patients in test_set/edge_cases.csv and checks each result automatically.

Run:  python -m src.stress_test                  (our system, RAG v3)
      python -m src.stress_test --system llm_only  (compare a baseline)
Writes: results/stress_test_<system>.csv  and  reports/week3_error_log.md
The edge cases are separate from the dev/test patients, so running them does not touch the locked test set.
"""
import argparse
import json
import time
from types import SimpleNamespace

import pandas as pd

from . import config, systems
from .data import drugs_in_text, normalize_drug, patient_record, suggest_drug
from .llm import LLMError

EDGE_CASES = config.TEST_SET / "edge_cases.csv"
BRANDS_TO_TEST = ["Coumadin", "Advil", "Motrin", "Aleve", "Plavix", "Zocor", "Lipitor", "Crestor", "Lasix", "Glucophage",
                  "Eliquis", "Xarelto", "Pradaxa", "Lanoxin", "Cordarone", "Pacerone", "Prilosec", "Biaxin", "Diflucan",
                  "Bactrim", "Lopressor", "Toprol XL", "Norvasc", "Zestril", "Cozaar", "Aldactone", "Viagra", "Nitrostat",
                  "Imdur", "Ranexa", "Multaq", "Tikosyn", "Brilinta", "Effient", "Lovenox", "Jardiance", "Farxiga",
                  "Zoloft", "Prozac", "Tegretol", "Dilantin", "Rifadin", "Sporanox", "Cardizem", "Calan", "Coreg"]
MISSPELLINGS = {"warfrin": "warfarin", "asprin": "aspirin", "simvastatn": "simvastatin", "metfromin": "metformin",
                "clopidogrell": "clopidogrel", "hydroxyzine": None, "levothyroxine": None}


def _sets(cell):
    """'a + b; c' -> [frozenset({a, b}), frozenset({c})]"""
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [frozenset(drugs_in_text(part)) for part in cell.split(";") if part.strip()]


def check_case(case, out, by_id):
    """Returns (passed, list of reasons it failed)."""
    flags = {frozenset(f["drugs"]): f["severity"] for f in out["flags"]}
    problems = []
    for need in _sets(case.must_flag):
        if not any(need <= k for k in flags):
            problems.append(f"missed {' + '.join(sorted(need))}")
    for bad in _sets(case.must_not_flag):
        hit = [k for k in flags if bad <= k] if len(bad) > 1 else [k for k in flags if k == bad]
        if hit:
            problems.append(f"false alarm on {' + '.join(sorted(bad))}")
    if str(case.expect_no_flags).lower() == "yes" and flags:
        problems.append("gave warnings for a patient where none were expected")
    for name in [n.strip() for n in str(case.must_list_not_checked).split(";") if n.strip() and n != "nan"]:
        if not any(name.lower() in x.lower() for x in out.get("not_in_library", [])):
            problems.append(f"'{name}' not reported as 'not checked'")
    if isinstance(case.same_as, str) and case.same_as.strip() and case.same_as in by_id:
        other = {frozenset(f["drugs"]): f["severity"] for f in by_id[case.same_as]["flags"]}
        if other != flags:
            problems.append(f"different result from {case.same_as} (should be the same)")
    return (not problems), problems


def name_recognition():
    brands = [(b, normalize_drug(b)) for b in BRANDS_TO_TEST]
    spelled = [(w, normalize_drug(w), suggest_drug(w), right) for w, right in MISSPELLINGS.items()]
    return brands, spelled


def run(system="rag_v2"):
    cases = pd.read_csv(EDGE_CASES, encoding="utf-8-sig")
    retriever = None
    if system.startswith("rag"):
        from .retrieval import Retriever
        retriever = Retriever("hybrid")
    outs, rows = {}, []
    for c in cases.itertuples():
        print(f"  {c.case_id} {c.test}: ", end="", flush=True)
        patient = patient_record(SimpleNamespace(
            patient_id=c.case_id, age=c.age, sex=c.sex, weight_kg=c.weight_kg,
            serum_creatinine_mg_dl=c.serum_creatinine_mg_dl, pregnant=c.pregnant, conditions=c.conditions,
            allergies=c.allergies, medications=c.medications))
        t0 = time.time()
        try:
            if system == "keyword":
                out = systems.keyword_system(patient)
            elif system == "llm_only":
                out = systems.llm_only_system(patient)
            else:
                out = systems.rag_system(patient, retriever, variant={"rag": "v1", "rag_v2": "v2", "rag_v3": "v3"}[system])
        except LLMError as e:
            raise SystemExit(f"\n\nStopped: {e}\nFinished cases are saved; run the same command again to continue.")
        except Exception as e:  # one broken case must not stop the others: it is logged as a failure
            out = {"flags": [], "removed_flags": [], "not_in_library": [], "meta": {}, "error": f"{type(e).__name__}: {e}"}
        outs[c.case_id] = out
        passed, problems = check_case(c, out, outs)
        if out.get("error"):
            passed, problems = False, [out["error"]]
        took = out.get("meta", {}).get("usage", {}).get("seconds")
        rows.append({
            "case_id": c.case_id, "category": c.category, "test": c.test,
            "expected": "; ".join(x for x in [
                f"warn: {c.must_flag}" if isinstance(c.must_flag, str) else "",
                f"no warning on: {c.must_not_flag}" if isinstance(c.must_not_flag, str) else "",
                "no warnings at all" if str(c.expect_no_flags).lower() == "yes" else "",
                f"list as not checked: {c.must_list_not_checked}" if isinstance(c.must_list_not_checked, str) else "",
                f"same result as {c.same_as}" if isinstance(c.same_as, str) else ""] if x),
            "system_output": "; ".join(f"[{f['severity']}] {' + '.join(f['drugs'])}" for f in out["flags"]) or "no warnings",
            "not_checked": ", ".join(out.get("not_in_library", [])),
            "removed_by_citation_check": len(out.get("removed_flags", [])),
            "requests": out.get("meta", {}).get("requests", ""),
            "seconds": took if took is not None else "",
            "auto_check": "PASS" if passed else "FAIL",
            "problems": "; ".join(problems),
            "pharmacist_review": "",
        })
        print("PASS" if passed else "FAIL " + "; ".join(problems))
    return pd.DataFrame(rows), outs


def write_log(df, system, brands, spelled):
    config.REPORTS.mkdir(exist_ok=True)
    n_pass = (df.auto_check == "PASS").sum()
    by_cat = df.groupby("category").auto_check.apply(lambda s: f"{(s == 'PASS').sum()}/{len(s)}")
    lines = [
        "# Week 3 Stress-Test Error Analysis Log" if system == "rag_v2" else f"# Stress-Test Error Analysis Log ({system})",
        "",
        f"*Generated by `python -m src.stress_test --system {system}`. Synthetic edge cases only "
        f"(`test_set/edge_cases.csv`), separate from the dev and test patients. Model: "
        f"{config.LLM_PROVIDER}. Not for clinical use.*",
        "",
        f"**Result: {n_pass} of {len(df)} edge cases passed the automatic check.**",
        "",
        "## Part A: Medicine names (no AI)",
        "",
        f"- Common US brand names recognised: **{sum(1 for _, d in brands if d)} of {len(brands)}** "
        "(before the brand list was added in Week 3: 7 of 46).",
        "- Misspellings are **never auto-corrected** (a wrong guess could swap two different drugs); "
        "the app shows a suggestion instead and lists the medicine as not checked:",
        "",
        "| Typed | Recognised as | Suggestion shown | Correct suggestion? |",
        "|---|---|---|---|",
    ]
    for w, rec, sug, right in spelled:
        ok = "yes" if sug == right else ("yes (no suggestion)" if right is None and sug is None else "no")
        lines.append(f"| {w} | {rec or 'not recognised'} | {sug or 'none'} | {ok} |")
    lines += ["", "## Part B: Edge cases (automatic checks)", "",
              "| Category | Passed |", "|---|---|"] + [f"| {k} | {v} |" for k, v in by_cat.items()]
    lines += ["", "| ID | Category | What is tested | Expected | System output | Check | Problem |",
              "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.case_id} | {r.category} | {r.test} | {r.expected} | {r.system_output} | "
                     f"{'✅ PASS' if r.auto_check == 'PASS' else '❌ FAIL'} | {r.problems} |")
    fails = df[df.auto_check == "FAIL"]
    lines += ["", "## Errors to analyse", ""]
    if fails.empty:
        lines.append("No automatic check failed. Pharmacy members should still review each output (column "
                     "`pharmacist_review` in the CSV).")
    else:
        for r in fails.itertuples():
            lines.append(f"- **{r.case_id} ({r.category})**: {r.problems}. Output: {r.system_output}. "
                         "*Cause and fix: to be filled in after review.*")
    # hand-written analysis for this version: Week 3 = RAG v2, Week 4 = RAG v3
    notes = config.REPORTS / {"rag_v2": "week3_error_notes.md", "rag_v3": "week4_error_notes.md"}.get(system, "none.md")
    if notes.exists():   # hand-written analysis of the failures (kept when the log is re-generated)
        lines += ["", notes.read_text(encoding="utf-8").strip()]
    lines += ["", "## How to review", "",
              f"Open `results/stress_test_{system}.csv` in Google Sheets or Excel. For every row, a pharmacy member "
              "writes in `pharmacist_review` whether the output is clinically acceptable (OK / not OK + why). "
              "Automatic checks only test what we told them to; a pharmacist can spot problems they miss."]
    path = config.REPORTS / ("week3_error_log.md" if system == "rag_v2" else f"stress_test_log_{system}.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description="Edge-case and bias testing")
    ap.add_argument("--system", default="rag_v3", choices=["rag_v3", "rag_v2", "rag", "llm_only", "keyword"])
    args = ap.parse_args()
    brands, spelled = name_recognition()
    print(f"Part A: brand names recognised {sum(1 for _, d in brands if d)}/{len(brands)}")
    print(f"Part B: {args.system} on {EDGE_CASES.name}")
    df, outs = run(args.system)
    config.RESULTS.mkdir(exist_ok=True)
    df.to_csv(config.RESULTS / f"stress_test_{args.system}.csv", index=False, encoding="utf-8-sig")
    with open(config.RESULTS / f"stress_test_{args.system}.jsonl", "w", encoding="utf-8") as fh:
        for cid, out in outs.items():
            fh.write(json.dumps({"case_id": cid, **out}, default=str) + "\n")
    path = write_log(df, args.system, brands, spelled)
    print(f"\n{(df.auto_check == 'PASS').sum()} of {len(df)} passed. Log written: {path}")


if __name__ == "__main__":
    main()
