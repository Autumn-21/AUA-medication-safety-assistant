"""Step 4 - Score the systems against the practice patients ("offline validation").

Run (from the project folder):
  python -m src.evaluate --system keyword                 # no API key needed
  python -m src.evaluate --system llm_only                # needs ANTHROPIC_API_KEY
  python -m src.evaluate --system rag                     # Week 2 version (v1); needs the AI model
  python -m src.evaluate --system rag_v2                  # Week 3 guardrails (v2); needs the AI model
  python -m src.evaluate --system rag_v3                  # Week 4 final (v3); needs the AI model
  python -m src.evaluate --system all
  python -m src.evaluate --retrieval                      # search quality only (no API key)

Only the DEV patients are used unless you pass  --split test --final  (Week 4 only: the test set must stay
unseen while we improve the system, otherwise the final score is not honest).

How a warning is matched to the correct answer: same patient + same set of drugs
(e.g. {warfarin, aspirin}). Warnings listed in test_set/acceptable_extras.csv are not counted either way.
"""
import argparse
import json
from collections import Counter

import pandas as pd

from . import config
from .data import load_acceptable_extras, load_expected, load_patients, patient_record

SEVERITY_ORDER = {"moderate": 0, "major": 1, "contraindicated": 2}
RAG_VARIANT = {"rag": "v1", "rag_v2": "v2", "rag_v3": "v3"}


def match_patient(pred_flags, expected_rows, acceptable):
    """Returns (true positives [(pred, exp)], false positives [pred], false negatives [exp])."""
    remaining = list(expected_rows)
    tps, fps, seen = [], [], set()
    for f in pred_flags:
        key = frozenset(f["drugs"])
        if not key or key in seen:          # empty or duplicate warning for the same drugs: ignore
            continue
        seen.add(key)
        hit = next((e for e in remaining if e["drug_set"] == key), None)
        if hit is not None:
            remaining.remove(hit)
            tps.append((f, hit))
        elif key in acceptable:
            continue
        else:
            fps.append(f)
    return tps, fps, remaining


_STOPWORDS = {"the", "and", "with", "for", "that", "this", "are", "was", "patients", "use", "when", "should", "not"}


def _content_words(text):
    from .citation_check import normalize
    return {w for w in normalize(text).split() if len(w) > 2 and w not in _STOPWORDS}


def citation_ok(flag, exp, gold, strict=False):
    """Is the warning backed by the right evidence?

    exact (strict=True): a verified quote from a chunk containing the expected evidence sentence,
                         or from the expected label section.
    relevant (default):  exact, OR
        - drug pair: a verified quote from either drug's label that names the other drug or its class
          (e.g. the fluconazole label saying it raises warfarin levels is valid evidence too)
        - single drug: a verified quote from that drug's label sharing most words with the expected evidence
    """
    from .citation_check import normalize
    from .data import drug_classes, patterns
    for c in flag["citations"]:
        if not c.get("verified"):
            continue
        if c.get("chunk_id") in gold.get(exp["flag_id"], []) or (
                c.get("drug") == exp["source_drug"] and c.get("field") == exp["source_section"]):
            return True
        if strict:
            continue
        drugs = sorted(exp["drug_set"])
        quote = normalize(c.get("quote", ""))
        if len(drugs) == 2 and c.get("drug") in drugs:
            partner = drugs[1] if c["drug"] == drugs[0] else drugs[0]
            names_partner = patterns()[partner].search(quote) or any(
                normalize(cls) in quote for cls in drug_classes().get(partner, []))
            if names_partner:
                return True
        if len(drugs) == 1 and c.get("drug") == drugs[0]:
            need = _content_words(exp["evidence_phrase"])
            if need and len(need & _content_words(quote)) / len(need) >= 0.6:
                return True
    return False


def score(predictions, split):
    patients = load_patients(split)
    expected = load_expected(split)
    acceptable = load_acceptable_extras()
    from .retrieval import gold_chunks
    gold = gold_chunks(expected)
    by_pid = {p["patient_id"]: p for p in predictions}

    rows, errors = [], []
    for pat in patients.itertuples():
        pred = by_pid[pat.patient_id]
        exp = expected[expected.patient_id == pat.patient_id].to_dict("records")
        tps, fps, fns = match_patient(pred["flags"], exp, acceptable.get(pat.patient_id, []))
        for f, e in tps:
            rows.append({"patient_id": pat.patient_id, "category": pat.category, "outcome": "TP",
                         "flag_type": e["flag_type"], "drugs": " + ".join(sorted(e["drug_set"])),
                         "severity_expected": e["severity"], "severity_given": f["severity"],
                         "citation_correct": citation_ok(f, e, gold),
                         "citation_exact": citation_ok(f, e, gold, strict=True),
                         "citation_verified": any(c.get("verified") for c in f["citations"])})
        for f in fps:
            rows.append({"patient_id": pat.patient_id, "category": pat.category, "outcome": "FP",
                         "flag_type": f["flag_type"], "drugs": " + ".join(sorted(f["drugs"])),
                         "severity_given": f["severity"], "explanation": f.get("explanation", ""),
                         "citation_verified": any(c.get("verified") for c in f["citations"])})
        for e in fns:
            rows.append({"patient_id": pat.patient_id, "category": pat.category, "outcome": "FN",
                         "flag_type": e["flag_type"], "drugs": " + ".join(sorted(e["drug_set"])),
                         "severity_expected": e["severity"]})
    df = pd.DataFrame(rows)

    def prf(sub_tp, sub_fp, sub_fn):
        p = sub_tp / (sub_tp + sub_fp) if sub_tp + sub_fp else 0.0
        r = sub_tp / (sub_tp + sub_fn) if sub_tp + sub_fn else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}

    count = Counter(df.outcome) if len(df) else Counter()
    tp, fp, fn = count["TP"], count["FP"], count["FN"]
    tp_rows = df[df.outcome == "TP"] if len(df) else df
    neg = patients[patients.n_expected_flags == 0].patient_id
    neg_with_fp = df[(df.outcome == "FP") & df.patient_id.isin(neg)].patient_id.nunique() if len(df) else 0

    all_flags = [f for p in predictions for f in p["flags"]]
    removed = [f for p in predictions for f in p.get("removed_flags", [])]
    raw_total = len(all_flags) + len(removed)
    unverified_shown = sum(1 for f in all_flags if not any(c.get("verified") for c in f["citations"]))

    grounded_tp = int(tp_rows.citation_correct.sum()) if tp else 0
    g_p = grounded_tp / (tp + fp) if tp + fp else 0.0
    g_r = grounded_tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "split": split, "patients": int(len(patients)), "expected_flags": int(len(expected)),
        "TP": tp, "FP": fp, "FN": fn, **prf(tp, fp, fn),
        # grounded = the warning is right AND its quoted evidence supports it
        "grounded_TP": grounded_tp, "grounded_precision": round(g_p, 4), "grounded_recall": round(g_r, 4),
        "grounded_f1": round(2 * g_p * g_r / (g_p + g_r), 4) if g_p + g_r else 0.0,
        "by_type": {},
        "false_alarm_patients_among_safe": f"{neg_with_fp}/{len(neg)}",
        "severity_exact": round((tp_rows.severity_expected == tp_rows.severity_given).mean(), 3) if tp else None,
        "citation_accuracy": round(tp_rows.citation_correct.mean(), 3) if tp else None,
        "citation_accuracy_exact": round(tp_rows.citation_exact.mean(), 3) if tp else None,
        "warnings_shown_without_verified_source": f"{unverified_shown}/{len(all_flags)}",
        "warnings_removed_by_citation_check": f"{len(removed)}/{raw_total}",
    }
    for t in ["interaction", "contraindication", "dose"]:
        t_tp = int(((df.outcome == "TP") & (df.flag_type == t)).sum()) if len(df) else 0
        t_fn = int(((df.outcome == "FN") & (df.flag_type == t)).sum()) if len(df) else 0
        # false positives are counted by the type the SYSTEM gave the warning
        t_fp = int(((df.outcome == "FP") & (df.flag_type == t)).sum()) if len(df) else 0
        t_g = int(((df.outcome == "TP") & (df.flag_type == t) & (df.citation_correct == True)).sum()) if tp else 0
        metrics["by_type"][t] = {"TP": t_tp, "FP": t_fp, "FN": t_fn, "grounded_TP": t_g, **prf(t_tp, t_fp, t_fn)}
    usage = [p["meta"].get("usage") for p in predictions if p["meta"].get("usage")]
    if usage:
        metrics["tokens_in"] = sum(u["input_tokens"] for u in usage)
        metrics["tokens_out"] = sum(u["output_tokens"] for u in usage)
        metrics["model"] = next((p["meta"].get("model") for p in predictions if p["meta"].get("model")), None)
    if any("repaired" in p["meta"] for p in predictions):
        metrics["warnings_kept_by_quote_repair"] = sum(p["meta"].get("repaired", 0) for p in predictions)
        metrics["flags_dismissed_by_self_check"] = sum(p["meta"].get("dismissed", 0) for p in predictions)
    ctx = [p["meta"]["n_context_chunks"] for p in predictions if "n_context_chunks" in p["meta"]]
    if ctx:
        metrics["avg_context_chunks"] = round(sum(ctx) / len(ctx), 1)
    return metrics, df


def run_system(name, split, mode="hybrid", k=config.TOP_K, model=None):
    from . import systems
    retriever = None
    if name.startswith("rag"):
        from .retrieval import Retriever
        retriever = Retriever(mode)
    preds = []
    for row in load_patients(split).itertuples():
        p = patient_record(row)
        print(f"  {p['patient_id']}: ", end="", flush=True)
        if name == "keyword":
            out = systems.keyword_system(p)
        elif name == "llm_only":
            out = systems.llm_only_system(p, model=model)
        else:
            out = systems.rag_system(p, retriever, k=k, model=model, variant=RAG_VARIANT[name])
        preds.append(out)
        took = out["meta"].get("usage", {}).get("seconds")
        print(f"{len(out['flags'])} warning(s)" + (" (saved answer)" if out["meta"].get("cached") else
                                                   f" ({took:.0f}s)" if took is not None else "")
              + (f", {len(out['removed_flags'])} removed by citation check" if out["removed_flags"] else ""))
    return preds


def evaluate_retrieval(split, ks=(1, 3, 5), modes=("bm25", "vector", "hybrid")):
    """Search quality: for each expected warning, is a chunk containing its evidence among the chunks
    retrieved for that patient?  (No AI involved.)"""
    from .retrieval import Retriever, gold_chunks, retrieve_for_patient
    expected = load_expected(split)
    gold = gold_chunks(expected)
    patients = {r.patient_id: patient_record(r) for r in load_patients(split).itertuples()}
    out = {}
    for mode in modes:
        r = Retriever(mode)
        for k in ks:
            hits, sizes, missed = 0, [], []
            for pid, p in patients.items():
                ids, _ = retrieve_for_patient(p, r, k=k)
                sizes.append(len(ids))
                for e in expected[expected.patient_id == pid].itertuples():
                    if set(gold[e.flag_id]) & set(ids):
                        hits += 1
                    else:
                        missed.append(f"{e.flag_id} {pid} {e.drugs}")
            out[f"{mode}@{k}"] = {"recall": round(hits / len(expected), 3), "found": hits,
                                  "expected": len(expected), "avg_chunks_per_patient": round(sum(sizes) / len(sizes), 1)}
            out[f"{mode}@{k}"]["missed"] = missed
    out["embedding_backend"] = (config.INDEXES / "backend.txt").read_text().strip()
    return out


def main():
    ap = argparse.ArgumentParser(description="Score the systems on the practice patients")
    ap.add_argument("--system", choices=["keyword", "llm_only", "rag", "rag_v2", "rag_v3", "all"],
                    help="rag = Week 2 (v1); rag_v2 = Week 3 guardrails; rag_v3 = Week 4 final")
    ap.add_argument("--retrieval", action="store_true", help="score the search step only")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="required for --split test (Week 4 only)")
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"], help="search type for rag")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="chunks kept per search")
    ap.add_argument("--model", default=None, help="AI model name (default: from .env / src/config.py)")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("The test patients are locked until the final evaluation (Week 4). "
                         "Use --split dev, or add --final if this IS the final run.")
    config.RESULTS.mkdir(exist_ok=True)

    if args.retrieval:
        res = evaluate_retrieval(args.split)
        (config.RESULTS / f"retrieval_{args.split}.json").write_text(json.dumps(res, indent=1))
        for name, v in res.items():
            if isinstance(v, dict):
                print(f"{name:12s} recall {v['recall']:.0%} ({v['found']}/{v['expected']}), "
                      f"{v['avg_chunks_per_patient']} chunks/patient")
    names = ["keyword", "llm_only", "rag", "rag_v2", "rag_v3"] if args.system == "all" else ([args.system] if args.system else [])
    from .llm import LLMError
    for name in names:
        print(f"\n== {name} on {args.split} patients ==")
        try:
            preds = run_system(name, args.split, args.mode, args.k, args.model)
        except LLMError as e:
            raise SystemExit(f"\n\nStopped: {e}")
        metrics, details = score(preds, args.split)
        if name.startswith("rag"):
            metrics["retrieval_mode"], metrics["k"] = args.mode, args.k
        tag = f"{name}_{args.split}"
        with open(config.RESULTS / f"predictions_{tag}.jsonl", "w", encoding="utf-8") as fh:
            for p in preds:
                fh.write(json.dumps(p, default=str) + "\n")
        details.to_csv(config.RESULTS / f"details_{tag}.csv", index=False)
        (config.RESULTS / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=1))
        print(f"precision {metrics['precision']:.2f}  recall {metrics['recall']:.2f}  F1 {metrics['f1']:.2f}  |  "
              f"grounded F1 {metrics['grounded_f1']:.2f}  citation accuracy {metrics['citation_accuracy']}  |  "
              f"false alarms on safe patients {metrics['false_alarm_patients_among_safe']}")
    if names or args.retrieval:
        print("\nSaved to results/. Build the report with:  python -m src.report")


if __name__ == "__main__":
    main()
