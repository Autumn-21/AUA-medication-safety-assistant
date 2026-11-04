# Medication Safety Assistant (Track A)

> ⚠️ **Not for clinical use.** This is a course prototype for the AI in Pharmacy Hackathon. It supports
> pharmacists and does not replace professional judgement. All patients in this project are synthetic (made up).

A retrieval-augmented (RAG) assistant that checks a patient's medicine list and flags:

1. **Drug–drug interactions**, e.g. warfarin + aspirin (bleeding risk)
2. **Contraindications**, e.g. metformin with severe kidney disease
3. **Dose problems**, e.g. apixaban 5 mg for an 85-year-old who weighs 52 kg

Every warning quotes the exact sentence from the official FDA drug label it relies on, with a link to the label on
DailyMed. Our code checks that each quote really appears in the label; a warning whose quote cannot be verified is
never shown as a confirmed warning.

**Final result on 22 locked test patients:** 17 of 18 problems found, and no warning shown without a verified label
quote (grounded F1 0.89, vs 0.11 for the same AI model answering without the label). Cost to run: $0 (free AI model).

## How it works

```
Patient + medicines
   │
   ├─ 1. Our code: recognise medicine names (incl. brand names), calculate CrCl and eGFR
   ├─ 2. Our code: hybrid search (BM25 keywords + vector search) over 2,478 FDA label chunks
   ├─ 3. AI model: read only the retrieved excerpts, write warnings as JSON with quotes
   ├─ 4. AI model: self-check — is each warning really a problem for this patient?
   └─ 5. Our code: citation check — the quote must be found word for word in the cited excerpt
            ├─ found      → warning shown with severity, advice, quote and DailyMed link
            └─ not found  → one repair try; if it still fails, shown only as "could not be verified"
```

| Part            | Model / tool                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------- |
| Language model  | Gemini 3.1 Flash Lite (Google AI Studio free tier); Ollama (local) also supported            |
| Embedding model | `BAAI/bge-small-en-v1.5` (open source, runs locally)                                         |
| Search          | BM25 (`rank-bm25`) + ChromaDB vectors, combined with reciprocal rank fusion, top 5 per query |
| Kidney function | Cockcroft–Gault CrCl and CKD-EPI 2021 eGFR, calculated in code (never by the AI)             |
| App             | Streamlit                                                                                    |

No model was retrained: RAG gives the model the right label text at the moment it answers.

## Results

**Final test** (22 locked test patients, never used during development; 18 expected warnings, 5 safe patients):

| System                            | F1       | Grounded F1 | Warnings without a verified source | False alarms on safe patients |
| --------------------------------- | -------- | ----------- | ---------------------------------- | ----------------------------- |
| Keyword checker (baseline, no AI) | 0.72     | 0.72        | 0 of 22                            | 3 of 5                        |
| AI only, no label (baseline)      | 0.94     | 0.11        | 16 of 18                           | 0 of 5                        |
| RAG v1 (Week 2)                   | 0.97     | 0.92        | 0 of 20                            | 1 of 5                        |
| RAG v2 (Week 3, guardrails)       | 0.91     | 0.85        | 0 of 16                            | 0 of 5                        |
| **RAG v3 (Week 4, final)**        | **0.94** | **0.89**    | **0 of 19**                        | 1 of 5                        |

_Grounded F1_ counts a warning only if its quote really supports it in the label. The three RAG versions are
within about one warning of each other, which is normal run-to-run variation for a small free model.

**Stress test:** RAG v3 passed all 21 edge cases (`test_set/edge_cases.csv`): brand names and misspellings, the
same patient as man/woman and aged 25/80 (identical results), prompt injection, kidney extremes and 8 medicines at once.

Full details: [final test report](reports/final_test_report.md) ·
[development results](reports/week2_benchmark_report.md) · [stress test log](reports/stress_test_log_rag_v3.md) ·
[Week 3 error log](reports/week3_error_log.md)

## Setup

You need **Python 3.10–3.12** and **Git**. Commands are for Windows cmd; macOS/Linux differences are in comments.

```bash
# 1. Get the project
git clone https://github.com/Autumn-21/AUA-medication-safety-assistant.git
cd AUA-medication-safety-assistant

# 2. Create and switch on a virtual environment
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate

# 3. Install the libraries (about 1-2 GB, includes PyTorch)
python -m pip install -r requirements.txt

# 4. Add your own free Gemini key
copy .env.example .env              # macOS/Linux: cp .env.example .env
#    open .env, paste your key from https://aistudio.google.com/apikey after GEMINI_API_KEY=
#    and keep GEMINI_MODEL=gemini-3.1-flash-lite (the model our saved results use)

# 5. Build the search indexes (downloads the embedding model once, ~130 MB)
python -m src.index

# 6. Check that everything works (no API key needed)
python -m pytest                    # should end with "25 passed"
```

Never upload `.env`; it is listed in `.gitignore`.

## Run the app

```bash
python -m streamlit run app.py
```

It opens at http://localhost:8501. Load a practice patient from the sidebar (for example **P033**) or type your own,
then press **Check medicines**. The sidebar can switch between our system and the two baselines. Stop with Ctrl + C.

Always start it with `python -m streamlit`, so the app uses the same Python the libraries were installed into.

## Other commands

| What                                          | Command                                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Check one patient from the command line       | `python -m src.run --patient P001`                                                            |
| Score a system on the practice (dev) patients | `python -m src.evaluate --system rag_v3` (also `keyword`, `llm_only`, `rag`, `rag_v2`, `all`) |
| Score search quality (Recall@k)               | `python -m src.evaluate --retrieval`                                                          |
| Run the 21 stress-test cases                  | `python -m src.stress_test --system rag_v3`                                                   |
| Rebuild the reports                           | `python -m src.report` (dev) · `python -m src.report --split test` (final test)               |
| Re-download the FDA labels                    | `python download_data.py` (about 3–5 minutes)                                                 |
| Re-cut the labels into chunks                 | `python -m src.chunking`                                                                      |

Every AI answer is saved in `results/cache/`, so re-running the same patients uses no new requests and gives
identical results. The final test (`--split test --final`) has been run once; running it again after more changes
would no longer be a fair test.

## Project structure

```
├── app.py                 # Streamlit web app
├── src/                   # the system (see src/README.md)
│   ├── config.py          #   settings, read from .env
│   ├── data.py            #   patients, medicine-name matching, brand names
│   ├── kidney.py          #   CrCl and eGFR formulas
│   ├── chunking.py        #   label sections -> 250-word chunks
│   ├── index.py           #   builds the BM25 and ChromaDB indexes
│   ├── retrieval.py       #   targeted hybrid search per drug pair and per drug
│   ├── prompts.py         #   instructions and JSON formats for the AI
│   ├── llm.py             #   free AI model calls (Gemini / Ollama), retries, cache
│   ├── systems.py         #   keyword baseline, AI-only baseline, RAG v1/v2/v3
│   ├── citation_check.py  #   verifies every quote against the label text
│   ├── evaluate.py        #   scores (precision, recall, F1, grounded F1, citations)
│   ├── stress_test.py     #   edge-case and bias testing
│   ├── report.py          #   writes the reports
│   └── run.py             #   check one patient from the command line
├── data/                  # 63 medicines, 769 FDA label sections, 2,478 chunks, drug classes, brand names
├── test_set/              # 44 synthetic patients (22 dev / 22 test), 37 expected warnings, 21 edge cases
├── notebooks/             # Week 1 exploratory data analysis (.ipynb and .html)
├── reports/               # benchmark, error logs, final test report, final report, presentation
├── results/               # scores, predictions and saved AI answers (cache/)
├── tests/                 # automated tests (python -m pytest)
├── download_data.py       # downloads and cleans the FDA labels (openFDA + RxNorm)
├── requirements.txt
└── .env.example           # template for your settings (copy to .env)
```

Not in the repository (re-create them): `indexes/` (`python -m src.index`), `data/raw/` (`python download_data.py`),
`.env` (your own key) and the demo video (too large for GitHub).

## Data sources

All sources are free and public.

| Source                                                                                           | Used for                                                  |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| [openFDA drug labels](https://open.fda.gov/apis/drug/label/)                                     | Official label text: the search library and the citations |
| [RxNorm API](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html) (NLM)                        | Standard drug names and IDs (RxCUI)                       |
| [DailyMed](https://dailymed.nlm.nih.gov/)                                                        | Links to the full official label for every citation       |
| [ONCHigh list](https://pmc.ncbi.nlm.nih.gov/articles/PMC3422823) (Phansalkar et al., JAMIA 2012) | High-risk drug pairs used to design test patients         |

## Limitations

- Small test set: 22 synthetic patients and 18 expected warnings; one warning moves F1 by about 0.03.
- 63 mostly cardiovascular medicines; any other medicine is listed as "not checked".
- A small free model: answers vary a little between runs, and severity for some warfarin interactions is one level
  lower than the expected answer.
- Synthetic patients written by the team; not validated by an independent clinical panel.

## Progress

| Week | Deliverables                                                                        | Status  |
| ---- | ----------------------------------------------------------------------------------- | ------- |
| 1    | Project proposal, EDA, data pipeline, test set, repository                          | ✅ Done |
| 2    | Search index, baselines vs RAG, citation check, benchmark report                    | ✅ Done |
| 3    | Guardrails, Streamlit app, edge-case and bias testing, error log                    | ✅ Done |
| 4    | Final fine-tuning (RAG v3), clinical review, final test, report, presentation, demo | ✅ Done |

## Team

| Name   | Role   |
| ------ | ------ |
| [Name] | [Role] |
| [Name] | [Role] |
| [Name] | [Role] |
| [Name] | [Role] |
