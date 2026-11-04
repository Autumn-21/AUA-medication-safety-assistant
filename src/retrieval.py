"""Step 3 - Search: find the label chunks relevant to one patient.

For every patient, a list of targeted searches is planned (one per drug pair and per drug), and each search
is restricted to the right label and sections. Example for warfarin + fluconazole:
  * in the WARFARIN label, interaction sections: "fluconazole ... azole antifungal CYP2C9 inhibitor ..."
  * in the FLUCONAZOLE label, interaction sections: "warfarin ... anticoagulant ..."
  * in each label, contraindication sections: the patient's conditions, allergies, pregnancy
  * in each label, dosage sections: kidney function, age, weight, dose

Searching by drug CLASS (from data/drug_classes.csv) is what finds sentences like
"strong CYP3A4 inhibitors" that never name the partner drug (EDA finding 7).
"""
import pickle
from functools import lru_cache
from itertools import permutations

import numpy as np
import pandas as pd

from . import config
from .data import SYNONYMS, drug_classes
from .index import get_embedder, load_chunks, tokenize, collection_name


@lru_cache(maxsize=1)
def _chunks():
    return load_chunks().set_index("chunk_id", drop=False)


def get_chunk(chunk_id):
    c = _chunks()
    return c.loc[chunk_id].to_dict() if chunk_id in c.index else None


class Retriever:
    """mode = 'bm25' (keywords), 'vector' (meaning) or 'hybrid' (both, merged by reciprocal-rank fusion)."""

    def __init__(self, mode="hybrid", backend=None):
        self.mode = mode
        with open(config.INDEXES / "bm25.pkl", "rb") as fh:
            saved = pickle.load(fh)
        self.bm25, self.ids = saved["bm25"], np.array(saved["chunk_ids"])
        chunks = _chunks().loc[self.ids]
        self.drug = chunks.drug.values
        self.field = chunks.field.values
        if mode in ("vector", "hybrid"):
            import chromadb
            if backend is None:
                backend = (config.INDEXES / "backend.txt").read_text().strip()
            self.backend = backend
            self.embedder = get_embedder(backend)
            client = chromadb.PersistentClient(path=str(config.INDEXES / "chroma"))
            self.collection = client.get_collection(collection_name(backend))

    def _bm25(self, query, drug, fields, n):
        scores = self.bm25.get_scores(tokenize(query))
        mask = (self.drug == drug) & np.isin(self.field, fields)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        best = idx[np.argsort(-scores[idx])][:n]
        return [(str(self.ids[i]), float(scores[i])) for i in best]

    def _vector(self, query, drug, fields, n):
        where = {"$and": [{"drug": drug}, {"field": {"$in": list(fields)}}]}
        res = self.collection.query(query_embeddings=[self.embedder.embed_query(query).tolist()],
                                    n_results=n, where=where)
        return [(cid, 1 - d) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def search(self, query, drug, fields, k=config.TOP_K):
        if self.mode == "bm25":
            return self._bm25(query, drug, fields, k)
        if self.mode == "vector":
            return self._vector(query, drug, fields, k)
        # hybrid: reciprocal-rank fusion of both result lists
        fused = {}
        for results in (self._bm25(query, drug, fields, 20), self._vector(query, drug, fields, 20)):
            for rank, (cid, _) in enumerate(results):
                fused[cid] = fused.get(cid, 0) + 1 / (config.RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


def _names(drug):
    return " ".join(dict.fromkeys([drug] + SYNONYMS.get(drug, [])))


def plan_queries(patient: dict):
    """The searches to run for one patient (see module docstring)."""
    classes = drug_classes()
    drugs = patient["in_scope_drugs"]
    dose_of = {m["drug"]: f"{m['dose']} {m['frequency']}" for m in patient["medications"] if m["drug"]}
    queries = []
    for a, b in permutations(drugs, 2):
        queries.append({
            "purpose": "interaction", "drug": a, "partner": b, "fields": config.SECTIONS_INTERACTION,
            "query": f"{_names(b)} {' '.join(classes.get(b, []))} concomitant use coadministration interaction "
                     f"contraindicated avoid increase risk dose"})
    pregnant = not str(patient["pregnant"]).lower().startswith("no")
    for d in drugs:
        queries.append({
            "purpose": "contraindication", "drug": d, "partner": None, "fields": config.SECTIONS_CONTRAINDICATION,
            "query": f"contraindicated in patients with {patient['conditions']} {patient['allergies']} "
                     f"hypersensitivity {'pregnancy pregnant fetal toxicity' if pregnant else ''}"})
        queries.append({
            "purpose": "dose", "drug": d, "partner": None, "fields": config.SECTIONS_DOSE,
            "query": f"recommended dosage {dose_of.get(d, '')} dose adjustment renal impairment creatinine "
                     f"clearance CrCl {patient['crcl_ml_min']} mL/min eGFR elderly age {patient['age']} "
                     f"body weight {patient['weight_kg']} kg"})
    return queries


def retrieve_for_patient(patient: dict, retriever: Retriever, k=config.TOP_K):
    """Run every planned search. Returns (unique chunk ids in order, per-query results)."""
    per_query, seen = [], {}
    for q in plan_queries(patient):
        hits = retriever.search(q["query"], q["drug"], q["fields"], k)
        per_query.append({**q, "hits": [cid for cid, _ in hits]})
        for cid, _ in hits:
            seen.setdefault(cid, None)
    return list(seen), per_query


def gold_chunks(expected: pd.DataFrame):
    """For each expected flag: the chunk ids that contain its evidence phrase (the 'right answer' for search)."""
    from .citation_check import normalize
    chunks = _chunks()
    out = {}
    for f in expected.itertuples():
        sub = chunks[(chunks.drug == f.source_drug) & (chunks.field == f.source_section)]
        phrase = normalize(f.evidence_phrase)
        out[f.flag_id] = [c for c, t in zip(sub.chunk_id, sub.text) if phrase in normalize(t)]
    return out
