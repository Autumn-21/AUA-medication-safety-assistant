"""Step 1 - Cut label sections into overlapping ~250-word chunks.

Every chunk keeps its drug, section and DailyMed link, so any warning built on it can be cited.

Run:  python -m src.chunking
"""
import pandas as pd

from . import config


def split_words(text: str, size: int, overlap: int):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return pieces


def build_chunks(size=config.CHUNK_WORDS, overlap=config.CHUNK_OVERLAP) -> pd.DataFrame:
    sections = pd.read_csv(config.LABEL_SECTIONS)
    rows = []
    for s in sections.itertuples():
        for i, piece in enumerate(split_words(str(s.text), size, overlap)):
            rows.append({
                "chunk_id": f"{s.drug}::{s.field}::{i:02d}",
                "drug": s.drug,
                "field": s.field,
                "section": config.SECTION_NAMES.get(s.field, s.field),
                "chunk_no": i,
                "text": piece,
                "n_words": len(piece.split()),
                "set_id": s.set_id,
                "dailymed_url": s.dailymed_url,
            })
    return pd.DataFrame(rows)


def main():
    chunks = build_chunks()
    chunks.to_csv(config.CHUNKS, index=False)
    print(f"{len(chunks):,} chunks from {chunks.drug.nunique()} drugs -> {config.CHUNKS}")
    print(f"median {int(chunks.n_words.median())} words per chunk")


if __name__ == "__main__":
    main()
