"""Project-wide settings. Change values here (or in .env) rather than inside the code."""
import os
from pathlib import Path

try:  # load API keys and settings from .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEST_SET = ROOT / "test_set"
INDEXES = ROOT / "indexes"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
CACHE = RESULTS / "cache"

LABEL_SECTIONS = DATA / "label_sections.csv"
DRUGS = DATA / "drugs.csv"
CHUNKS = DATA / "chunks.csv"
DRUG_CLASSES = DATA / "drug_classes.csv"
PATIENTS = TEST_SET / "patients.csv"
EXPECTED_FLAGS = TEST_SET / "expected_flags.csv"
ACCEPTABLE_EXTRAS = TEST_SET / "acceptable_extras.csv"

# --- Chunking -------------------------------------------------------------
CHUNK_WORDS = 250      # words per chunk (EDA: 57% of sections are longer than this)
CHUNK_OVERLAP = 50     # words shared by neighbouring chunks, so no sentence is lost at a boundary

# --- Search ---------------------------------------------------------------
# "sentence-transformers" = real meaning-based embeddings (needs internet once to download the model)
# "tfidf" = offline fallback (word statistics + SVD); weaker, used only when the model can't be downloaded
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.getenv("TOP_K", "5"))          # chunks kept per search (5 found 100% of evidence on dev)
RRF_K = 60                                    # constant for reciprocal-rank fusion (hybrid search)
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "40"))  # above this, review one medicine at a time (v2)

# --- AI model -------------------------------------------------------------
# gemini = Google Gemini free tier (default) | ollama = free model on your laptop | anthropic = Claude (paid)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")   # a free "flash" model; list yours: python -m src.llm
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_SECONDS_BETWEEN_CALLS = float(os.getenv("LLM_SECONDS_BETWEEN_CALLS", "13"))  # free tier: ~5 requests/minute
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))           # give up on one request after this
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_OUTPUT_TOKENS = 4000

# Label sections grouped by what they are used for. Old-format labels use "warnings" + "precautions",
# new-format labels use "warnings_and_cautions" (EDA finding 4), so both are listed.
SECTIONS_INTERACTION = ["drug_interactions", "contraindications", "boxed_warning", "warnings_and_cautions",
                        "warnings", "precautions", "dosage_and_administration",
                        "do_not_use", "ask_doctor_or_pharmacist"]
SECTIONS_CONTRAINDICATION = ["contraindications", "boxed_warning", "warnings_and_cautions", "warnings",
                             "precautions", "use_in_specific_populations", "pregnancy", "do_not_use",
                             "ask_doctor", "pregnancy_or_breast_feeding"]
SECTIONS_DOSE = ["dosage_and_administration", "use_in_specific_populations", "geriatric_use",
                 "precautions", "warnings_and_cautions", "warnings"]

SECTION_NAMES = {
    "boxed_warning": "Boxed Warning", "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration", "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions", "warnings": "Warnings", "precautions": "Precautions",
    "drug_interactions": "Drug Interactions", "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy", "pediatric_use": "Pediatric Use", "geriatric_use": "Geriatric Use",
    "adverse_reactions": "Adverse Reactions", "overdosage": "Overdosage",
    "clinical_pharmacology": "Clinical Pharmacology", "do_not_use": "Do Not Use",
    "ask_doctor": "Ask a Doctor Before Use", "ask_doctor_or_pharmacist": "Ask a Doctor or Pharmacist Before Use",
    "stop_use": "Stop Use", "pregnancy_or_breast_feeding": "Pregnancy or Breast-feeding",
}
