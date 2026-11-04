# src/: system code

The pipeline, in the order it runs:

| File | Step | What it does |
|---|---|---|
| `config.py` | Settings | Paths, chunk size, search settings, AI model; reads `.env` |
| `chunking.py` | 1. Prepare | Cuts label sections into ~250-word chunks (50-word overlap) → `data/chunks.csv` |
| `data.py` | Features | Loads patients; matches drug names ("diltiazem ER" → diltiazem); drug classes from `data/drug_classes.csv` |
| `kidney.py` | Features | Creatinine clearance (Cockcroft-Gault) and eGFR (CKD-EPI 2021), in code, never by the AI |
| `index.py` | 2. Index | Builds the keyword index (BM25) and the vector index (ChromaDB) → `indexes/` |
| `retrieval.py` | 3. Search | Plans targeted searches per drug pair and per drug; keyword / vector / hybrid search |
| `prompts.py` | 4. AI | Instructions and JSON output format for the AI |
| `llm.py` | 4. AI | Calls a free AI model (Gemini free tier or Ollama; Claude optional) with JSON output; saves every answer in `results/cache/` |
| `systems.py` | 4. AI | The three systems: keyword baseline, AI-only baseline, RAG (v1 Week 2, v2 Week 3, v3 Week 4 final: self-check + quote repair) |
| `citation_check.py` | 5. Guardrail | Removes any warning whose quote is not really in the cited label text |
| `evaluate.py` | 6. Score | Precision, recall, F1, grounded F1, citation accuracy, false alarms (dev patients only) |
| `report.py` | 7. Report | Writes `reports/week2_benchmark_report.md` from `results/` |
| `stress_test.py` | Week 3–4 | Edge-case and bias testing; writes `reports/week3_error_log.md` (v2) or `reports/stress_test_log_rag_v3.md` (v3) |
| `run.py` | Use it | Check one patient from the command line |
| `../app.py` | Use it | Web app (Streamlit): `python -m streamlit run app.py` |
