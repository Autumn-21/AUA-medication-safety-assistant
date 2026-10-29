# src/: system code (Week 2 onwards)

Planned modules, one per step of the pipeline:

| File | Step | What it does |
|---|---|---|
| `chunking.py` | Prepare | Cut label sections into ~250-word pieces, keeping drug, section and DailyMed link |
| `normalize.py` | Match names | Brand/generic names to RxCUI (RxNorm) and drug classes (RxClass) |
| `retrieval.py` | Search | Hybrid search (keyword BM25 + embeddings) over the label pieces |
| `kidney.py` | Calculate | Creatinine clearance and eGFR in code (never by the AI) |
| `generate.py` | AI warnings | Send patient + retrieved pieces to the AI; get warnings as JSON with citations |
| `citation_check.py` | Verify | Delete any warning whose quote is not in the cited label piece |
| `evaluate.py` | Score | Precision, recall, citation accuracy on the dev (later test) patients |
| `app.py` | Show | Streamlit web page for pharmacists |
