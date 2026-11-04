"""Prompts and output formats for the AI systems."""

SEVERITY_RULES = """Severity (use the label's own wording):
- "contraindicated": the label says contraindicated, do not use, or must not be used.
- "major": the label says avoid, reduce or change the dose, do not exceed a dose, or not recommended.
- "moderate": the label says monitor, use caution, or consider a dose change."""

SCOPE_RULES = """Flag only problems that apply to THIS patient:
1. interaction: two of the patient's medicines that the label warns about when used together
   (including dose limits for a combination, e.g. "do not exceed X mg with drug Y").
2. contraindication: a medicine unsafe for this patient because of a condition, pregnancy, allergy or kidney function.
3. dose: a dose that does not match the label for this patient's kidney function (CrCl / eGFR given),
   age or body weight.
Do not flag general warnings that apply to every patient taking the drug unless a feature of this patient or one
of their other medicines triggers them. Use the CrCl and eGFR values given; do not recalculate them.
Medicines marked [NOT IN OUR LABEL LIBRARY] must not be flagged; list them in not_in_library.
If there is nothing to flag, return an empty flags list."""

RAG_SYSTEM_V1 = f"""You are a clinical pharmacy decision-support assistant that reviews a patient's medicines for a
pharmacist. You must work ONLY from the FDA label excerpts provided with each request.

Grounding rules (most important):
- Every claim must be supported by the excerpts. Do not use outside knowledge for any warning.
- Every flag must cite at least one excerpt by its exact id, with a quote of 8 to 40 consecutive words copied
  exactly from that excerpt (same words, same order).
- If the excerpts do not support a concern, do not flag it.

{SCOPE_RULES}

{SEVERITY_RULES}"""

# Version 2 (Week 3): extra rules written from the errors on DEV patients only (see reports/).
CHECK_RULES_V2 = """Before you flag anything, check these rules:
- Quotes: copy ONE continuous passage from a single excerpt, word for word. Do not join sentences from different
  places or change any word.
- Dose limits: when a label gives a maximum or recommended dose (alone, with another drug, or for a level of kidney
  function, age or weight), compare it with the patient's ACTUAL dose. Flag only if the patient's dose is above the
  limit or does not match what the label recommends for this patient. A combination the label allows at the
  patient's current dose is not a problem.
- Pregnancy: flag a medicine only if its label says it is contraindicated in pregnancy, should be avoided or
  discontinued, or has a boxed warning about harm to the fetus. Statements such as "use only if the potential
  benefit justifies the potential risk" are not flags.
- Check every medicine: for each one, look for interactions with each other medicine, contraindications for this
  patient, and whether the dose fits this patient's kidney function, age and weight."""

RAG_SYSTEM = RAG_SYSTEM_V1 + "\n\n" + CHECK_RULES_V2

LLM_ONLY_SYSTEM = f"""You are a clinical pharmacy decision-support assistant that reviews a patient's medicines for a
pharmacist, using your knowledge of the official FDA drug labels (no documents are provided).

For every flag, name the label section the information comes from and quote the label's wording as exactly as you can
(8 to 40 words).

{SCOPE_RULES}

{SEVERITY_RULES}"""

_FLAG_BASE = {
    "flag_type": {"type": "string", "enum": ["interaction", "contraindication", "dose"]},
    "drugs": {"type": "array", "items": {"type": "string"},
              "description": "Drug names exactly as given in 'Drug names to use'"},
    "severity": {"type": "string", "enum": ["contraindicated", "major", "moderate"]},
    "explanation": {"type": "string", "description": "One or two sentences for the pharmacist"},
    "recommendation": {"type": "string", "description": "What the label says to do"},
}


def _schema(citation_props, base=None):
    base = base or _FLAG_BASE
    citation = {"type": "object", "properties": citation_props,
                "required": list(citation_props), "additionalProperties": False}
    flag = {"type": "object",
            "properties": {**base, "citations": {"type": "array", "items": citation}},
            "required": list(base) + ["citations"], "additionalProperties": False}
    return {"type": "object",
            "properties": {"flags": {"type": "array", "items": flag},
                           "not_in_library": {"type": "array", "items": {"type": "string"}}},
            "required": ["flags", "not_in_library"], "additionalProperties": False}


RAG_SCHEMA = _schema({"chunk_id": {"type": "string"}, "quote": {"type": "string"}})

# --- Version 3 (Week 4 final fine-tuning), written from the Week 3 stress-test errors (E10, E11) -----------------
SELF_CHECK_V3 = """- Self-check every candidate flag: in "explanation", first compare the patient's ACTUAL situation (their dose,
  kidney function, age, weight, other medicines, pregnancy, conditions) with what the label requires. Then set
  "is_problem": true only if the label shows the patient's current medicines or dose are a problem; false if the
  label shows they are acceptable (for example, the dose already matches the label's recommendation for this
  patient). Only flags with is_problem = true are shown to the pharmacist.
- Quotes: copy the label's sentence exactly as written, keeping its numbers as digits and its word order."""

RAG_SYSTEM_V3 = RAG_SYSTEM + "\n" + SELF_CHECK_V3

_FLAG_BASE_V3 = {
    "flag_type": _FLAG_BASE["flag_type"],
    "drugs": _FLAG_BASE["drugs"],
    "explanation": {"type": "string", "description": "Compare the patient's actual situation with the label, "
                                                     "in one or two sentences for the pharmacist"},
    "is_problem": {"type": "boolean", "description": "true only if the label shows this is a problem for this patient"},
    "severity": _FLAG_BASE["severity"],
    "recommendation": _FLAG_BASE["recommendation"],
}
RAG_SCHEMA_V3 = _schema({"chunk_id": {"type": "string"}, "quote": {"type": "string"}}, base=_FLAG_BASE_V3)

# Repair step: when a warning's quote is not found in its excerpt, ask once more for an exact copy.
REPAIR_SYSTEM = """You copy text exactly. You are given ONE excerpt from an FDA drug label and a warning that cites it.
Copy, word for word, the shortest continuous passage (8 to 40 words) of the excerpt that supports the warning.
Do not change, reorder, shorten or add any word; keep numbers exactly as written. If no passage of the excerpt supports
the warning, set "found" to false and "quote" to an empty string."""
REPAIR_SCHEMA = {"type": "object",
                 "properties": {"found": {"type": "boolean"}, "quote": {"type": "string"}},
                 "required": ["found", "quote"], "additionalProperties": False}


def repair_user(flag, chunk):
    return (f"WARNING\n{flag.get('flag_type')}: {' + '.join(flag.get('drugs', []))}\n{flag.get('explanation', '')}\n\n"
            f'EXCERPT\n<excerpt id="{chunk["chunk_id"]}" drug="{chunk["drug"]}" section="{chunk["section"]}">\n'
            f'{chunk["text"]}\n</excerpt>\n\nCopy the supporting passage exactly.')
LLM_ONLY_SCHEMA = _schema({"drug_label": {"type": "string"}, "section": {"type": "string"},
                           "quote": {"type": "string"}})


def format_excerpts(chunks):
    return "\n\n".join(
        f'<excerpt id="{c["chunk_id"]}" drug="{c["drug"]}" section="{c["section"]}">\n{c["text"]}\n</excerpt>'
        for c in chunks)


def rag_user(patient_text, drug_names, chunks, focus=None):
    task = ("Review this patient's medicines and return the flags." if not focus else
            f"Review only problems that involve {focus} (its interactions with each other medicine, its "
            f"contraindications for this patient, and its dose). The other medicines are reviewed separately.")
    return (f"PATIENT\n{patient_text}\n\nDrug names to use: {', '.join(drug_names) or '(none)'}\n\n"
            f"FDA LABEL EXCERPTS ({len(chunks)})\n{format_excerpts(chunks)}\n\n{task}")


def llm_only_user(patient_text, drug_names):
    return (f"PATIENT\n{patient_text}\n\nDrug names to use: {', '.join(drug_names) or '(none)'}\n\n"
            "Review this patient's medicines and return the flags.")
