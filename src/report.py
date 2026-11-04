"""Step 5 - Write the benchmarking report from the saved results.

Run:  python -m src.report                 -> reports/week2_benchmark_report.md  (dev patients)
      python -m src.report --split test    -> reports/final_test_report.md       (locked test patients, Week 4)
Re-run it after every evaluation; the numbers are always taken from results/.
"""
import json
from datetime import date

import pandas as pd

from . import config

SYSTEMS = [("keyword", "Baseline 1: keyword checker"), ("llm_only", "Baseline 2: AI only (no search)"),
           ("rag", "RAG v1 (Week 2)"), ("rag_v2", "RAG v2 (Week 3 guardrails)"), ("rag_v3", "RAG v3 (Week 4 final)")]


def _load(split):
    out = {}
    for name, _ in SYSTEMS:
        p = config.RESULTS / f"metrics_{name}_{split}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def _pct(x):
    return "–" if x is None else f"{x:.0%}" if isinstance(x, float) else str(x)


def _num(x):
    return "–" if x is None else f"{x:.2f}"


def build(split="dev"):
    m = _load(split)
    retr_path = config.RESULTS / f"retrieval_{split}.json"
    retr = json.loads(retr_path.read_text()) if retr_path.exists() else None
    lines = [
        "# Week 2 Benchmarking Report – Medication Safety Assistant (Track A)" if split == "dev" else
        "# Final Test Report (Week 4) – Medication Safety Assistant (Track A)",
        "",
        f"*Generated {date.today().isoformat()} by `python -m src.report` from the files in `results/`. "
        + (f"Patients: **{split}** split only (the test split stays locked until Week 4).*" if split == "dev" else
         "Patients: the 22 locked **test** patients, used once for the final evaluation after all tuning was "
         "finished.*"),
        "",
        "> Not for clinical use. All patients are synthetic.",
        "",
        "## 1. What we compared",
        "",
        "| System | How it works |",
        "|---|---|",
        "| Baseline 1: keyword checker | Flags two drugs if one's name appears in the other's warning sections; "
        "flags a contraindication if a patient condition word appears in a label's Contraindications/Boxed Warning. No AI. |",
        "| Baseline 2: AI only | The AI model reviews the patient from memory, with no label text given. "
        "Its quotes are then searched for in the real labels. |",
        "| RAG v1 (Week 2) | Targeted search of the FDA labels (per drug pair and per drug, restricted to the right "
        "label and sections, using drug classes) → the AI answers **only** from the retrieved excerpts → "
        "citation check removes any warning whose quote is not in the cited excerpt. |",
        "| RAG v2 (Week 3) | v1 + stricter rules (one continuous quote, compare the actual dose with the label limit, "
        "pregnancy flagged only if contraindicated/avoid/boxed warning, check every medicine) + patients with more than "
        f"{config.MAX_CONTEXT_CHUNKS} excerpts reviewed one medicine at a time. |",
        "| RAG v3 (Week 4) | v2 + self-check (the AI states whether each flag is really a problem for this patient; "
        "\"no\" is dropped) + quote repair (a quote that fails the check gets one more request to copy it exactly). |",
        "",
        "## 2. Main results",
        "",
    ]
    n_exp = next((v["expected_flags"] for v in m.values()), "?")
    n_pat = next((v["patients"] for v in m.values()), "?")
    lines += [f"{n_pat} dev patients, {n_exp} expected warnings. A warning counts as correct when it names the same "
              "drugs as the expected one; **grounded** means it is correct *and* its quoted evidence supports it.", ""]
    head = "| Score | " + " | ".join(label for _, label in SYSTEMS) + " |"
    lines += [head, "|---" * (len(SYSTEMS) + 1) + "|"]

    def row(label, key, fmt=_num):
        cells = [fmt(m[n].get(key)) if n in m else "not run yet" for n, _ in SYSTEMS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Precision (warnings given that were correct)", "precision")
    row("Recall (expected warnings found)", "recall")
    row("**F1**", "f1")
    row("**Grounded F1** (correct + supported by its quote)", "grounded_f1")
    row("Citation accuracy (correct warnings with supporting evidence)", "citation_accuracy", _pct)
    row("Citation accuracy, strict (exact expected sentence/section)", "citation_accuracy_exact", _pct)
    row("Severity matches the expected severity", "severity_exact", _pct)
    row("Safe patients with a false alarm", "false_alarm_patients_among_safe", str)
    row("Warnings shown without a verified source", "warnings_shown_without_verified_source", str)
    row("Warnings removed by the citation check", "warnings_removed_by_citation_check", str)
    lines += ["", "### By warning type (F1; found / expected)", "",
              "| Type | " + " | ".join(label for _, label in SYSTEMS) + " |", "|---" * (len(SYSTEMS) + 1) + "|"]
    for t in ["interaction", "contraindication", "dose"]:
        cells = []
        for n, _ in SYSTEMS:
            if n not in m:
                cells.append("not run yet")
                continue
            b = m[n]["by_type"][t]
            cells.append(f"{b['f1']:.2f} ({b['TP']}/{b['TP'] + b['FN']}, {b['FP']} false alarm{'' if b['FP'] == 1 else 's'})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    if retr:
        lines += ["", "## 3. Search quality (no AI involved)", "",
                  f"Embedding backend: `{retr.get('embedding_backend')}`. Recall = share of expected warnings whose "
                  "evidence sentence was among the chunks retrieved for that patient (k = chunks kept per search).", "",
                  "| Search | k=1 | k=3 | k=5 | chunks per patient (k=5) |", "|---|---|---|---|---|"]
        for mode, label in [("bm25", "Keyword (BM25)"), ("vector", "Vector (embeddings)"), ("hybrid", "Hybrid (both)")]:
            if f"{mode}@5" in retr:
                lines.append(f"| {label} | " + " | ".join(_pct(retr[f'{mode}@{k}']['recall']) for k in (1, 3, 5))
                             + f" | {retr[f'{mode}@5']['avg_chunks_per_patient']} |")
        missed = retr.get("hybrid@3", {}).get("missed", [])
        if missed:
            lines += ["", "Evidence missed by hybrid search at k=3: " + "; ".join(missed) + "."]

    lines += ["", "## 4. Errors to learn from", ""]
    for n, label in SYSTEMS:
        p = config.RESULTS / f"details_{n}_{split}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        fn = d[d.outcome == "FN"]
        fp = d[d.outcome == "FP"]
        lines += [f"**{label}**", ""]
        lines.append("- Missed: " + ("; ".join(f"{r.patient_id} {r.drugs} ({r.flag_type})" for r in fn.itertuples())
                                     or "none"))
        lines.append("- False alarms: " + ("; ".join(f"{r.patient_id} {r.drugs} ({r.flag_type})"
                                                     for r in fp.itertuples()) or "none"))
        lines.append("")

    findings = config.REPORTS / ("week2_findings_dev.md" if split == "dev" else "final_test_findings.md")
    if findings.exists():   # hand-written interpretation of the numbers (kept when the report is re-generated)
        lines += [findings.read_text(encoding="utf-8").strip(), ""]

    cost = [(label, m[n]) for n, label in SYSTEMS if n in m and "tokens_in" in m[n]]
    if cost:
        lines += ["## 5. AI usage", "", "Free model (Gemini free tier or local Ollama): no cost.", "",
                  "| System | Model | Input tokens | Output tokens | Avg chunks sent |",
                  "|---|---|---|---|---|"]
        for label, v in cost:
            lines.append(f"| {label} | {v.get('model')} | {v['tokens_in']:,} | {v['tokens_out']:,} | "
                         f"{v.get('avg_context_chunks', '–')} |")
        lines.append("")

    lines += [
        "## 6. Method notes",
        "",
        "- **No data leakage:** prompts, search settings and the drug-class list are tuned on dev patients only. "
        "`src/evaluate.py` refuses to run on the test split without `--final`.",
        "- **Kidney function** (Cockcroft-Gault CrCl, CKD-EPI 2021 eGFR) is calculated in code and given to the AI.",
        "- **Matching:** same patient + same set of drugs. Duplicate warnings count once. Warnings listed in "
        "`test_set/acceptable_extras.csv` (e.g. intended dual antiplatelet therapy) are neither rewarded nor penalised.",
        "- **Citation check (guardrail):** a quote must be at least 4 words and appear in the cited excerpt; the RAG "
        "system may only cite excerpts it was given. Failing warnings are removed before display.",
        "- **Why F1 and not ROC-AUC:** our systems output a list of warnings, not a probability score per drug pair, "
        "so ROC-AUC does not apply. F1 (per warning type), citation accuracy and false alarms on safe patients are the "
        "scores that match the clinical task.",
        "- **Reproducible:** every AI answer is saved in `results/cache/`; re-running gives identical results at no cost.",
        "",
        "## 7. Reproduce",
        "",
        "```",
        "python -m src.chunking",
        "python -m src.index",
        "python -m src.evaluate --retrieval",
        "python -m src.evaluate --system all",
        "python -m src.report",
        "```",
    ]
    config.REPORTS.mkdir(exist_ok=True)
    out = config.REPORTS / ("week2_benchmark_report.md" if split == "dev" else "final_test_report.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write the benchmark report from results/")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    build(ap.parse_args().split)
