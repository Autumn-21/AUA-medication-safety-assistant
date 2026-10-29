# Medication Safety Assistant (Track A)

> ⚠️ **Not for clinical use.** This is a hackathon research prototype. It supports pharmacists and
> does not replace professional judgement. All patient data in this project is synthetic (made up).

An AI assistant that checks a patient's medicine list and flags:

1. **Drug–drug interactions** (two medicines that are risky together)
2. **Contraindications** (a medicine unsafe for this patient, e.g. pregnancy, asthma, allergy)
3. **Dose problems** (e.g. a dose too high for the patient's kidney function)

Every warning cites the exact section of the official FDA drug label it comes from
(retrieval-augmented generation with grounded citations).

## Project structure

```
├── data/
│   ├── drugs.csv              # 63 medicines: chosen label, sections, DailyMed link
│   ├── label_sections.csv     # FDA label text, one row per section (the search library)
│   ├── missing_report.csv     # drugs/labels that need checking (empty = all good)
│   └── raw/                   # original openFDA downloads (not uploaded; re-create with the script)
├── test_set/
│   ├── patients.csv           # synthetic practice patients (dev / test split)
│   ├── expected_flags.csv     # correct warnings for each patient, with label evidence
│   ├── build_test_set.py      # makes the two CSV files above
│   ├── verify_test_set.py     # checks every expected warning is in the label text
│   └── README.md              # test-set guide and pharmacist review checklist
├── notebooks/
│   ├── week1_eda_analysis.ipynb   # Week 1 exploratory data analysis
│   └── week1_eda_analysis.html    # same report, opens in a browser
├── src/                       # system code (Week 2 onwards)
├── drug_list.csv              # the medicines in scope
├── download_data.py           # downloads and cleans FDA labels (openFDA + RxNorm)
├── requirements.txt           # Python libraries
└── .env.example               # template for API keys (copy to .env)
```

## Setup

You need **Python 3.10 or newer** and **Git**.

```bash
# 1. Get the project
git clone https://github.com/<your-team>/medication-safety-assistant.git
cd medication-safety-assistant

# 2. Create a virtual environment (keeps this project's libraries separate)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install the libraries
pip install -r requirements.txt

# 4. Add your API keys
copy .env.example .env          # Windows  (macOS/Linux: cp .env.example .env)
# then open .env and paste your keys. Never upload .env to GitHub.
```

## How to run

```bash
# Download and clean the FDA labels (about 3-5 minutes; creates data/)
python download_data.py

# Check that every expected warning in the test set is backed by label text
python test_set/verify_test_set.py      # should print 37/37

# Open the EDA notebook
jupyter notebook notebooks/week1_eda_analysis.ipynb
```

## Data sources

All sources are free and public; no licence is required.

| Source | Used for |
|---|---|
| [openFDA drug labels](https://open.fda.gov/apis/drug/label/) | Official label text: the search library and citations |
| [RxNorm API](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html) (NLM) | Standard drug names and IDs (RxCUI) |
| [DailyMed](https://dailymed.nlm.nih.gov/) | Links to the full official label for every citation |
| [ONCHigh list](https://pmc.ncbi.nlm.nih.gov/articles/PMC3422823) (Phansalkar et al., JAMIA 2012) | High-risk drug pairs used to design test patients |

## Evaluation

- `test_set/` holds synthetic patients split into **dev** (for building) and **test**
  (locked until the final evaluation in Week 4).
- Metrics: precision, recall and F1 per warning type; retrieval Recall@k; citation accuracy;
  unsupported-claim rate.

## Progress

| Week | Deliverables | Status |
|---|---|---|
| 1 | Project proposal, EDA, data pipeline, test set, repository | ✅ Done |
| 2 | Retrieval (search) + AI warnings with citations, baselines | ⏳ Next |
| 3 | Kidney dose checks, web app, improvements on dev set | |
| 4 | Final evaluation on test set, demo, presentation | |

## Team

| Name | Role |
|---|---|
| [Name] | [e.g. Clinical lead / test set] |
| [Name] | [e.g. Data pipeline] |
| [Name] | [e.g. Retrieval and AI] |
| [Name] | [e.g. Evaluation and app] |
