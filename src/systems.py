"""The three systems compared in the benchmark.

  keyword   Baseline 1 - flags a pair if one drug's name appears in the other's warning sections. No AI.
  llm_only  Baseline 2 - the AI answers from memory, with no label text given.
  rag       Our system - targeted search -> AI answers only from the retrieved excerpts -> citation check.

Each system returns the same format:
  {"patient_id", "flags": [ {flag_type, drugs, severity, explanation, recommendation, citations[...]} ],
   "removed_flags": [...flags that failed the citation check...], "not_in_library": [...], "meta": {...}}
"""
import re
from itertools import combinations

from . import config, prompts
from .citation_check import find_quote_in_labels, verify_citation
from .data import normalize_drug, patient_summary, patterns
from .llm import call_json
from .retrieval import _chunks, get_chunk, retrieve_for_patient

SENTENCE = re.compile(r"(?<=[.;•])\s+")


_TYPES = {"interaction", "contraindication", "dose"}
_SEVERITIES = {"contraindicated", "major", "moderate"}


def _clean_flag(f):
    """Free / smaller models sometimes bend the format a little; make every flag look the same."""
    f = dict(f) if isinstance(f, dict) else {}
    t = str(f.get("flag_type", "")).lower()
    f["flag_type"] = next((x for x in _TYPES if x in t), "interaction" if len(f.get("drugs") or []) > 1 else "dose")
    sev = str(f.get("severity", "")).lower()
    f["severity"] = next((x for x in _SEVERITIES if x in sev), "moderate")
    drugs = f.get("drugs") or []
    f["drugs"] = [drugs] if isinstance(drugs, str) else list(drugs)
    cits = f.get("citations") or []
    f["citations"] = [c for c in (cits if isinstance(cits, list) else [cits]) if isinstance(c, dict)]
    f.setdefault("explanation", ""), f.setdefault("recommendation", "")
    return f


def _flags_from(output):
    flags = output.get("flags", []) if isinstance(output, dict) else []
    return [_clean_flag(f) for f in flags if isinstance(f, dict)]


def _normalize_drugs(names, allowed):
    out = []
    for n in names:
        d = normalize_drug(n)
        if d and d in allowed and d not in out:
            out.append(d)
    return out


# --- Baseline 1: keyword ----------------------------------------------------
KEYWORD_IX_FIELDS = ["drug_interactions", "contraindications", "boxed_warning", "do_not_use",
                     "ask_doctor_or_pharmacist"]
KEYWORD_CI_FIELDS = ["contraindications", "boxed_warning", "do_not_use"]
_STOP = {"history", "chronic", "acute", "placed", "stable", "after", "months", "weeks", "years", "treatment",
         "community", "acquired", "infection", "surgery", "prophylaxis", "documented", "none", "started"}


def _first_sentence_with(chunks, pattern):
    for c in chunks.itertuples():
        for s in SENTENCE.split(c.text):
            if pattern.search(s):
                words = s.split()
                return c, " ".join(words[:40])
    return None, None


def keyword_system(patient: dict) -> dict:
    chunks = _chunks()
    drugs = patient["in_scope_drugs"]
    pats = patterns()
    flags = []
    for a, b in combinations(drugs, 2):
        for x, y in ((a, b), (b, a)):
            sub = chunks[(chunks.drug == x) & chunks.field.isin(KEYWORD_IX_FIELDS)]
            c, sentence = _first_sentence_with(sub, pats[y])
            if c is not None:
                flags.append({"flag_type": "interaction", "drugs": [a, b],
                              "severity": "contraindicated" if "contraindicat" in sentence.lower() else "major",
                              "explanation": f"{y} is mentioned in the {c.section} section of the {x} label.",
                              "recommendation": "Review the label section.",
                              "citations": [{"chunk_id": c.chunk_id, "quote": sentence}]})
                break
    pregnant = not patient["pregnant"].lower().startswith("no")
    words = {w for phrase in re.split(r"[;,()]", f"{patient['conditions']};{patient['allergies']}")
             for w in re.findall(r"[a-z]{6,}", phrase.lower()) if w not in _STOP}
    for d in drugs:
        sub = chunks[(chunks.drug == d) & chunks.field.isin(KEYWORD_CI_FIELDS)]
        terms = sorted(words) + (["pregnan"] if pregnant else [])
        if not terms:
            continue
        pat = re.compile("|".join(terms), re.I)
        c, sentence = _first_sentence_with(sub, pat)
        if c is not None:
            flags.append({"flag_type": "contraindication", "drugs": [d],
                          "severity": "contraindicated" if "contraindicat" in sentence.lower() else "major",
                          "explanation": f"A patient condition is mentioned in the {c.section} section.",
                          "recommendation": "Review the label section.",
                          "citations": [{"chunk_id": c.chunk_id, "quote": sentence}]})
    for f in flags:
        f["citations"] = [verify_citation(c) for c in f["citations"]]
    return {"patient_id": patient["patient_id"], "flags": flags, "removed_flags": [],
            "not_in_library": patient["out_of_scope"], "meta": {"system": "keyword"}}


# --- Baseline 2: AI only (no search) ------------------------------------------
def llm_only_system(patient: dict, model=None) -> dict:
    user = prompts.llm_only_user(patient_summary(patient), patient["in_scope_drugs"])
    res = call_json(prompts.LLM_ONLY_SYSTEM, user, prompts.LLM_ONLY_SCHEMA, model=model)
    flags = []
    for f in _flags_from(res["output"]):
        f["drugs"] = _normalize_drugs(f["drugs"], patient["in_scope_drugs"])
        # The AI was given no text, so check whether its quote exists anywhere in these drugs' labels.
        f["citations"] = [find_quote_in_labels(str(c.get("quote", "")), f["drugs"] or patient["in_scope_drugs"])
                          | {"claimed_section": c.get("section")} for c in f["citations"]]
        flags.append(f)
    return {"patient_id": patient["patient_id"], "flags": flags, "removed_flags": [],
            # what was checked is decided by our own name matching, never by the AI (Week 3 stress test, E01/E12)
            "not_in_library": patient["out_of_scope"],
            "meta": {"system": "llm_only", "model": res["model"], "usage": res["usage"], "cached": res["cached"]}}


# --- Our system: RAG -------------------------------------------------------------
def _ask(patient, chunk_ids, system_prompt, schema, model, focus=None):
    chunks = [get_chunk(c) for c in chunk_ids]
    user = prompts.rag_user(patient_summary(patient), patient["in_scope_drugs"], chunks, focus=focus)
    return call_json(system_prompt, user, schema, model=model)


def _is_problem(flag):
    v = flag.get("is_problem", True)
    return not (v is False or str(v).strip().lower() in ("false", "no", "0"))


def _repair(flag, allowed, model):
    """v3: one more request asking the AI to copy the supporting sentence exactly from the excerpt it cited.
    Returns (verified citation or None, list of request results)."""
    results = []
    for c in flag["citations"]:
        cid = c.get("chunk_id")
        if cid not in allowed:
            continue
        res = call_json(prompts.REPAIR_SYSTEM, prompts.repair_user(flag, get_chunk(cid)), prompts.REPAIR_SCHEMA,
                        model=model)
        results.append(res)
        out = res["output"] if isinstance(res["output"], dict) else {}
        if out.get("found") and out.get("quote"):
            fixed = verify_citation({"chunk_id": cid, "quote": out["quote"]}, allowed)
            if fixed["verified"]:
                return fixed, results
    return None, results


VARIANTS = {"v1": (prompts.RAG_SYSTEM_V1, prompts.RAG_SCHEMA),
            "v2": (prompts.RAG_SYSTEM, prompts.RAG_SCHEMA),
            "v3": (prompts.RAG_SYSTEM_V3, prompts.RAG_SCHEMA_V3)}


def rag_system(patient: dict, retriever, k=config.TOP_K, model=None, variant="v3") -> dict:
    """variant "v1" = Week 2 version (one request, original prompt).
    variant "v2" = Week 3 guardrails: stricter prompt rules, and patients whose search returns more than
    MAX_CONTEXT_CHUNKS excerpts are reviewed one medicine at a time (smaller, focused requests).
    variant "v3" = Week 4 final: v2 + self-check (the AI must state whether each flag is really a problem for this
    patient; "no" is dropped) + quote repair (a failed quote gets one more request to copy it exactly)."""
    chunk_ids, per_query = retrieve_for_patient(patient, retriever, k=k)
    system_prompt, schema = VARIANTS[variant]
    drugs = patient["in_scope_drugs"]

    if variant in ("v2", "v3") and len(chunk_ids) > config.MAX_CONTEXT_CHUNKS and len(drugs) > 1:
        requests = []
        for d in drugs:   # the excerpts found in THIS drug's label (its pairs, contraindications and dose)
            ids = list(dict.fromkeys(cid for q in per_query if q["drug"] == d for cid in q["hits"]))
            requests.append((d, ids))
    else:
        requests = [(None, chunk_ids)]

    kept, removed, dismissed, seen, results = [], [], [], set(), []
    for focus, ids in requests:
        res = _ask(patient, ids, system_prompt, schema, model, focus)
        results.append(res)
        allowed = set(ids)
        for f in _flags_from(res["output"]):
            f["drugs"] = _normalize_drugs(f["drugs"], drugs)
            if variant == "v3" and not _is_problem(f):
                dismissed.append(f)      # self-check: the AI itself concluded this is not a problem (Week 3, E10)
                continue
            f["citations"] = [verify_citation(c, allowed) for c in f["citations"]]
            if variant == "v3" and f["drugs"] and not any(c["verified"] for c in f["citations"]):
                fixed, repair_results = _repair(f, allowed, model)   # Week 3, E11
                results.extend(repair_results)
                if fixed:
                    f["citations"], f["repaired"] = [fixed], True
            # Guardrail: a warning is shown only if at least one of its quotes is really in a cited excerpt.
            if not (any(c["verified"] for c in f["citations"]) and f["drugs"]):
                removed.append(f)
                continue
            key = frozenset(f["drugs"])
            if key not in seen:          # the same pair can be found from both drugs' labels: keep it once
                seen.add(key)
                kept.append(f)

    usage = {"input_tokens": sum(r["usage"].get("input_tokens", 0) for r in results),
             "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in results),
             "seconds": round(sum(r["usage"].get("seconds", 0) for r in results), 1)}
    return {"patient_id": patient["patient_id"], "flags": kept, "removed_flags": removed,
            "dismissed_flags": dismissed,
            # what was checked is decided by our own name matching, never by the AI (Week 3 stress test, E01/E12)
            "not_in_library": patient["out_of_scope"],
            "meta": {"system": "rag" if variant == "v1" else f"rag_{variant}", "variant": variant,
                     "model": results[0]["model"], "usage": usage, "cached": all(r["cached"] for r in results),
                     "requests": len(results), "retrieval_mode": retriever.mode, "k": k,
                     "repaired": sum(1 for f in kept if f.get("repaired")), "dismissed": len(dismissed),
                     "context_chunks": chunk_ids, "n_context_chunks": len(chunk_ids)}}
