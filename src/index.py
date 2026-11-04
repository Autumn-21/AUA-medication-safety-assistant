"""Step 2 - Build the search indexes over the label chunks ("vector indexing").

Two indexes are built:
  * keyword index (BM25)      - good for exact drug names
  * vector index (ChromaDB)   - good for meaning ("bleeding risk" also finds "hemorrhage")

Run:  python -m src.index               (uses EMBEDDING_BACKEND from config / .env)
      python -m src.index --backend tfidf   (offline fallback, no model download)
"""
import argparse
import pickle
import re

import numpy as np
import pandas as pd

from . import config

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# Prefix recommended by the BGE embedding models for search queries.
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
                "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: "}


def tokenize(text: str):
    return TOKEN.findall(str(text).lower())


def search_text(row) -> str:
    """What gets indexed: drug + section name + the chunk text."""
    return f"{row.drug} - {row.section}: {row.text}"


def load_chunks() -> pd.DataFrame:
    if not config.CHUNKS.exists():
        raise SystemExit("data/chunks.csv not found. Run first:  python -m src.chunking")
    return pd.read_csv(config.CHUNKS)


# --- Embedding backends ----------------------------------------------------
class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name=config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # imported here: large library
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.prefix = QUERY_PREFIX.get(model_name, "")

    def embed_documents(self, texts):
        return self.model.encode(list(texts), batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=True).astype(np.float32)

    def embed_query(self, text):
        return self.model.encode([self.prefix + text], normalize_embeddings=True)[0].astype(np.float32)


class TfidfEmbedder:
    """Offline fallback: TF-IDF word statistics compressed with SVD (latent semantic analysis)."""
    name = "tfidf"
    model_name = "tfidf-svd-256"

    def __init__(self, state=None):
        self.vectorizer, self.svd = state if state else (None, None)

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        x = self.vectorizer.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=256, random_state=0).fit(x)
        return self

    def _embed(self, texts):
        v = self.svd.transform(self.vectorizer.transform(texts))
        return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)).astype(np.float32)

    def embed_documents(self, texts):
        return self._embed(list(texts))

    def embed_query(self, text):
        return self._embed([text])[0]


def collection_name(backend):
    return f"labels_{backend.replace('-', '_')}"


def get_embedder(backend):
    if backend == "tfidf":
        with open(config.INDEXES / "tfidf.pkl", "rb") as fh:
            return TfidfEmbedder(pickle.load(fh))
    return SentenceTransformerEmbedder()


def build(backend: str):
    import chromadb

    config.INDEXES.mkdir(exist_ok=True)
    chunks = load_chunks()
    texts = [search_text(r) for r in chunks.itertuples()]

    # 1. Keyword index (BM25)
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(config.INDEXES / "bm25.pkl", "wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": chunks.chunk_id.tolist()}, fh)
    print(f"Keyword index: {len(texts):,} chunks")

    # 2. Vector index (ChromaDB)
    if backend == "tfidf":
        embedder = TfidfEmbedder().fit(texts)
        with open(config.INDEXES / "tfidf.pkl", "wb") as fh:
            pickle.dump((embedder.vectorizer, embedder.svd), fh)
    else:
        embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed_documents(texts)

    client = chromadb.PersistentClient(path=str(config.INDEXES / "chroma"))
    name = collection_name(backend)
    if name in [c.name if hasattr(c, "name") else c for c in client.list_collections()]:
        client.delete_collection(name)
    col = client.create_collection(name, metadata={"hnsw:space": "cosine", "model": embedder.model_name})
    for start in range(0, len(chunks), 500):
        part = chunks.iloc[start:start + 500]
        col.add(ids=part.chunk_id.tolist(),
                embeddings=vectors[start:start + 500].tolist(),
                documents=texts[start:start + 500],
                metadatas=[{"drug": r.drug, "field": r.field} for r in part.itertuples()])
    print(f"Vector index: {col.count():,} chunks, backend={backend}, model={embedder.model_name}")
    with open(config.INDEXES / "backend.txt", "w") as fh:
        fh.write(backend)


def main():
    ap = argparse.ArgumentParser(description="Build the search indexes")
    ap.add_argument("--backend", default=config.EMBEDDING_BACKEND, choices=["sentence-transformers", "tfidf"])
    build(ap.parse_args().backend)


if __name__ == "__main__":
    main()
