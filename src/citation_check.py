"""Guardrail - check that every citation is real.

A warning passes only if its quote can actually be found in the label chunk it cites
(after ignoring case, spacing and punctuation differences). The quote may leave words out, but every
piece of it must be copied exactly, in order. Warnings that fail are removed from the final answer,
so the system cannot present made-up evidence.
"""
import re

from .retrieval import _chunks, get_chunk

_PUNCT = re.compile(r"[^a-z0-9%<>≤≥=./+ -]+")
_SENTENCE_PUNCT = re.compile(r"(?<!\d)[.,;:]|[.,;:](?!\d)")   # sentence punctuation, but keep 12.3 or 0.5
_ELLIPSIS = re.compile(r"\.\.\.|…")
MIN_PIECE_WORDS = 4     # every copied piece must be at least this long
MAX_PIECES = 4          # a quote may skip text at most 3 times


def normalize(text: str) -> str:
    text = str(text).lower().replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    text = _ELLIPSIS.sub(" ", text)
    text = _SENTENCE_PUNCT.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str, min_words=MIN_PIECE_WORDS) -> bool:
    """True if the quote is copied from the text.

    Allowed: the quote is ONE exact passage, or a few exact passages of the text in their original order
    with words skipped between them (e.g. a "[see section 8.7]" reference left out).
    Not allowed: any word that is not in the text at that point, pieces shorter than 4 words, or pieces
    out of order. So nothing can be invented; only left out.
    """
    q_words = normalize(quote).split()
    if len(q_words) < min_words:
        return False            # very short quotes ("aspirin") prove nothing
    t = " " + normalize(text) + " "
    pos, i, pieces = 0, 0, 0
    while i < len(q_words):
        found = None
        for j in range(len(q_words), i + min_words - 1, -1):     # longest exact piece starting at word i
            k = t.find(" " + " ".join(q_words[i:j]) + " ", pos)
            if k != -1:
                found = (j, k + len(" ".join(q_words[i:j])) + 1)
                break
        if found is None:
            return False
        i, pos = found
        pieces += 1
        if pieces > MAX_PIECES:
            return False
    return True


def verify_citation(citation: dict, allowed_chunk_ids=None) -> dict:
    """citation = {"chunk_id": ..., "quote": ...}. Adds 'verified' and the chunk's drug/section."""
    cid = citation.get("chunk_id")
    chunk = get_chunk(cid) if cid else None
    ok = bool(chunk) and quote_in_text(citation.get("quote", ""), chunk["text"])
    if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
        ok = False              # the AI may only cite chunks it was actually given
    return {**citation, "verified": ok,
            "drug": chunk["drug"] if chunk else None, "field": chunk["field"] if chunk else None,
            "dailymed_url": chunk["dailymed_url"] if chunk else None}


def find_quote_in_labels(quote: str, drugs) -> dict:
    """For the AI-only baseline (no chunks given): search the drugs' labels for the quoted text."""
    chunks = _chunks()
    sub = chunks[chunks.drug.isin(list(drugs))]
    for r in sub.itertuples():
        if quote_in_text(quote, r.text):
            return {"chunk_id": r.chunk_id, "quote": quote, "verified": True, "drug": r.drug,
                    "field": r.field, "dailymed_url": r.dailymed_url}
    return {"chunk_id": None, "quote": quote, "verified": False, "drug": None, "field": None, "dailymed_url": None}
