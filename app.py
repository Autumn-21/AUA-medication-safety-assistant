"""Medication Safety Assistant - web app (Week 3 MVP).

Run from the project folder:   python -m streamlit run app.py
("python -m" makes sure the app uses the same Python as the other commands.)
It opens in your browser at http://localhost:8501
"""
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from src import config, systems
from src.data import load_expected, load_patients, normalize_drug, patient_record, suggest_drug
from src.kidney import crcl_cockcroft_gault, egfr_ckd_epi_2021
from src.llm import LLMError

st.set_page_config(page_title="MediSafe: Medication Safety Assistant", page_icon="💊", layout="wide")

SEVERITY = {  # icon, label, how to show it
    "contraindicated": ("🔴", "CONTRAINDICATED", st.error),
    "major": ("🟠", "MAJOR", st.warning),
    "moderate": ("🟡", "MODERATE", st.info),
}
SEVERITY_ORDER = {"contraindicated": 0, "major": 1, "moderate": 2}
TYPE_NAME = {"interaction": "Drug–drug interaction", "contraindication": "Contraindication", "dose": "Dose problem"}
SYSTEMS = {
    "Our system (RAG v3, final)": "rag_v3",
    "Baseline: AI only (no label search)": "llm_only",
    "Baseline: keyword checker": "keyword",
}
EMPTY_MEDS = pd.DataFrame([{"Medicine": "", "Dose": "", "Frequency": ""}])


@st.cache_resource(show_spinner="Loading the FDA label search index…")
def get_retriever():
    from src.retrieval import Retriever
    return Retriever("hybrid")


@st.cache_data
def practice_patients():
    # Only DEV patients can be loaded: the test patients stay unseen until the final evaluation.
    return load_patients("dev").set_index("patient_id")


def meds_to_frame(med_string):
    rows = []
    for part in str(med_string).split(";"):
        bits = [b.strip() for b in part.split("|")] + ["", "", ""]
        if bits[0]:
            rows.append({"Medicine": bits[0], "Dose": bits[1], "Frequency": bits[2]})
    return pd.DataFrame(rows or EMPTY_MEDS.to_dict("records"))


def load_practice_patient():
    pid = st.session_state.practice
    if pid == "(none)":
        return
    p = practice_patients().loc[pid]
    st.session_state.update(age=int(p.age), sex=p.sex, weight=float(p.weight_kg),
                            scr=float(p.serum_creatinine_mg_dl),
                            pregnant="No" if str(p.pregnant).lower().startswith("no") else "Yes",
                            conditions=str(p.conditions), allergies=str(p.allergies))
    st.session_state.meds_df = meds_to_frame(p.medications)
    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
    st.session_state.pop("result", None)


def show_flag(f):
    icon, label, box = SEVERITY.get(f["severity"], ("⚪", f["severity"].upper(), st.info))
    with st.container(border=True):
        box(f"**{icon} {label} · {TYPE_NAME.get(f['flag_type'], f['flag_type'])}: {' + '.join(f['drugs'])}**")
        if f.get("explanation"):
            st.markdown(f["explanation"])
        if f.get("recommendation"):
            st.markdown(f"**What the label says to do:** {f['recommendation']}")
        for c in f["citations"]:
            section = config.SECTION_NAMES.get(c.get("field"), c.get("field") or c.get("claimed_section") or "")
            if c.get("verified"):
                st.markdown(f"✅ **Verified in the official label** — {c.get('drug')} label, *{section}*")
            else:
                st.markdown(f"⚠️ **Quote NOT found in the official label** (claimed: {section or 'unknown section'})")
            st.markdown("> " + str(c.get("quote", "")).replace("\n", " "))
            if c.get("dailymed_url"):
                st.markdown(f"[Open the full label on DailyMed]({c['dailymed_url']})")


# --- Page ------------------------------------------------------------------------------------
st.title("💊 MediSafe: Medication Safety Assistant")
st.caption("Checks a patient's medicines against official FDA drug labels: drug–drug interactions, "
           "contraindications and dose problems. Every warning quotes the label it comes from.")
st.warning("**Prototype for pharmacist decision support only — not for clinical use.** "
           "Use synthetic (made-up) patients only. Covers 63 medicines.", icon="⚠️")

with st.sidebar:
    st.header("Settings")
    system_label = st.radio("System", list(SYSTEMS), index=0,
                            help="Compare our system with the two baselines from the benchmark.")
    st.selectbox("Load a practice patient (dev set)", ["(none)"] + list(practice_patients().index),
                 key="practice", on_change=load_practice_patient)
    show_expected = st.checkbox("Show the expected answer for the practice patient", value=False)
    st.divider()
    st.caption(f"AI model: {config.LLM_PROVIDER} · "
               f"{config.GEMINI_MODEL if config.LLM_PROVIDER == 'gemini' else config.OLLAMA_MODEL}")
    st.caption("Kidney function is calculated in code (Cockcroft–Gault CrCl, CKD-EPI 2021 eGFR), not by the AI.")

DEFAULTS = {"age": 65, "sex": "F", "weight": 70.0, "scr": 1.0, "pregnant": "No", "conditions": "",
            "allergies": "none", "meds_df": EMPTY_MEDS.copy(), "editor_version": 0}
for key, value in DEFAULTS.items():      # set once; afterwards the widgets keep what the user typed
    st.session_state.setdefault(key, value)

left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("1. Patient")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", min_value=18, max_value=110, key="age")
    sex = c2.selectbox("Sex", ["F", "M"], key="sex")
    weight = c3.number_input("Weight (kg)", min_value=25.0, max_value=250.0, step=1.0, key="weight")
    c4, c5 = st.columns(2)
    scr = c4.number_input("Serum creatinine (mg/dL)", min_value=0.2, max_value=15.0, step=0.1, key="scr")
    pregnant = c5.selectbox("Pregnant", ["No", "Yes"], key="pregnant")
    conditions = st.text_input("Conditions", placeholder="e.g. atrial fibrillation; asthma", key="conditions")
    allergies = st.text_input("Allergies", key="allergies")

    k1, k2 = st.columns(2)
    k1.metric("CrCl (mL/min)", f"{crcl_cockcroft_gault(age, weight, scr, sex):.0f}",
              help="Creatinine clearance, Cockcroft–Gault, actual body weight. Calculated in code.")
    k2.metric("eGFR (mL/min/1.73m²)", f"{egfr_ckd_epi_2021(age, scr, sex):.0f}",
              help="CKD-EPI 2021 equation. Calculated in code.")

    st.subheader("2. Medicines")
    meds = st.data_editor(st.session_state.meds_df, num_rows="dynamic", use_container_width=True,
                          key=f"meds_editor_{st.session_state.editor_version}",
                          column_config={"Medicine": st.column_config.TextColumn(help="Generic or brand name"),
                                         "Dose": st.column_config.TextColumn(help="e.g. 5 mg"),
                                         "Frequency": st.column_config.TextColumn(help="e.g. once daily")})
    names = [str(m).strip() for m in meds["Medicine"].fillna("") if str(m).strip()]
    if names:
        def recognised(n):
            d = normalize_drug(n)
            if not d:
                hint = suggest_drug(n)
                return f"⚪ {n} (not recognised" + (f" — did you mean **{hint}**?)" if hint else "; not in our label library)")
            return f"✅ {n}" if d == n.strip().lower() else f"✅ {n} → {d}"
        st.caption("Recognised: " + " · ".join(recognised(n) for n in names))
    run = st.button("Check medicines", type="primary", use_container_width=True, disabled=not names)

with right:
    st.subheader("3. Warnings")
    if run:
        med_string = "; ".join(
            f"{r.Medicine} | {'' if pd.isna(r.Dose) else r.Dose} | {'' if pd.isna(r.Frequency) else r.Frequency}"
            for r in meds.itertuples() if str(r.Medicine).strip() and not pd.isna(r.Medicine))
        patient = patient_record(SimpleNamespace(
            patient_id=st.session_state.practice if st.session_state.practice != "(none)" else "NEW",
            age=age, sex=sex, weight_kg=weight, serum_creatinine_mg_dl=scr, pregnant=pregnant,
            conditions=conditions or "none", allergies=allergies or "none", medications=med_string))
        system = SYSTEMS[system_label]
        try:
            with st.spinner("Searching the labels and asking the AI model… (the free model can take up to a minute)"):
                if system == "keyword":
                    out = systems.keyword_system(patient)
                elif system == "llm_only":
                    out = systems.llm_only_system(patient)
                else:
                    out = systems.rag_system(patient, get_retriever(), variant="v3")
            st.session_state.result = {"out": out, "system": system_label}
        except LLMError as e:
            st.error(f"The AI model could not answer: {e}")

    result = st.session_state.get("result")
    if not result:
        st.info("Enter a patient and medicines (or load a practice patient from the sidebar), "
                "then press **Check medicines**.")
    else:
        out = result["out"]
        flags = sorted(out["flags"], key=lambda f: SEVERITY_ORDER.get(f["severity"], 9))
        counts = pd.Series([f["severity"] for f in flags]).value_counts().to_dict() if flags else {}
        summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda x: SEVERITY_ORDER.get(x[0], 9)))
        st.markdown(f"**{result['system']}** found **{len(flags)} warning{'s' if len(flags) != 1 else ''}**"
                    + (f" ({summary})." if summary else "."))
        if not flags:
            st.success("No problems found in the label evidence for these medicines. "
                       "This does not replace a pharmacist's review.")
        for f in flags:
            show_flag(f)
        if out.get("not_in_library"):
            st.info("**Not checked (not in our label library):** " + ", ".join(out["not_in_library"]))
        if out.get("removed_flags"):
            # Week 4: a possible problem whose quote could not be verified is NOT hidden; it is shown clearly as
            # unverified, so the pharmacist can check the label themselves (Week 3 stress test, E11).
            st.warning(f"**⚠️ {len(out['removed_flags'])} possible problem(s) could not be verified.** The AI raised "
                       "them, but its quote was not found word for word in the label, so they are not counted as "
                       "warnings. Please check the label yourself:")
            for f in out["removed_flags"]:
                drugs = " + ".join(f["drugs"]) or "(unknown medicines)"
                st.markdown(f"- **{drugs}** ({f.get('flag_type', '')}): {f.get('explanation', '')}")
        if out.get("dismissed_flags"):
            with st.expander(f"Self-check: {len(out['dismissed_flags'])} candidate(s) the AI checked and found "
                             "acceptable for this patient"):
                for f in out["dismissed_flags"]:
                    st.markdown(f"- {' + '.join(f['drugs']) or '(unknown)'}: {f.get('explanation', '')}")
        meta = out["meta"]
        with st.expander("Technical details"):
            st.write({k: meta[k] for k in ("system", "model", "requests", "n_context_chunks", "repaired", "dismissed",
                                           "cached", "usage")
                      if k in meta})

    pid = st.session_state.get("practice", "(none)")
    if show_expected and pid != "(none)":
        st.divider()
        st.subheader(f"Expected answer for {pid}")
        exp = load_expected("dev")
        exp = exp[exp.patient_id == pid]
        if exp.empty:
            st.success("No warning expected: this is a safe patient.")
        else:
            st.dataframe(exp[["flag_type", "drugs", "severity", "expected_message", "source_drug", "source_section"]],
                         hide_index=True, use_container_width=True)
        note = practice_patients().loc[pid].notes
        if isinstance(note, str) and note.strip():
            st.caption(f"Note: {note}")

st.divider()
st.caption("Data: openFDA drug labels, RxNorm, DailyMed. All patients are synthetic. "
           "Track A – Clinical Pharmacy & Patient Safety. Not for clinical use.")
