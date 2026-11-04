"""Tests for the Week 2 pipeline. Run:  python -m pytest -q

The AI is replaced by a fake model here, so the tests need no API key and cost nothing.
Requires the indexes:  python -m src.chunking  then  python -m src.index --backend tfidf
"""
import pytest

from src import data, prompts, systems
from src.chunking import split_words
from src.citation_check import quote_in_text
from src.evaluate import match_patient, score
from src.kidney import crcl_cockcroft_gault, egfr_ckd_epi_2021
from src.retrieval import Retriever, get_chunk, gold_chunks


def test_kidney_formulas():
    # (140-80)*70/(72*1.0) = 58.3 ; x0.85 for women = 49.6
    assert round(crcl_cockcroft_gault(80, 70, 1.0, "M"), 1) == 58.3
    assert round(crcl_cockcroft_gault(80, 70, 1.0, "F"), 1) == 49.6
    assert 25 < egfr_ckd_epi_2021(76, 1.9, "F") < 30


def test_drug_name_matching():
    assert data.normalize_drug("metoprolol succinate") == "metoprolol"
    assert data.normalize_drug("diltiazem ER") == "diltiazem"
    assert data.normalize_drug("Coumadin") == "warfarin"
    assert data.normalize_drug("albuterol inhaler") is None
    assert data.drugs_in_text("warfarin + sulfamethoxazole/trimethoprim") == {
        "warfarin", "sulfamethoxazole and trimethoprim"}


def test_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(600)]
    pieces = split_words(" ".join(words), 250, 50)
    assert [len(p.split()) for p in pieces] == [250, 250, 200]
    assert pieces[1].split()[0] == "w200"           # 50-word overlap
    assert pieces[-1].split()[-1] == "w599"         # nothing lost at the end


def test_quote_check():
    text = "Concomitant use of strong CYP3A4 inhibitors is contraindicated."
    assert quote_in_text("concomitant use of strong CYP3A4 inhibitors", text)
    assert not quote_in_text("concomitant use of strong CYP2D6 inhibitors", text)   # made-up quote
    assert not quote_in_text("CYP3A4", text)                                         # too short to prove anything


def test_matching_counts_duplicates_once():
    exp = [{"drug_set": frozenset({"warfarin", "aspirin"}), "flag_id": "F1"}]
    preds = [{"drugs": ["warfarin", "aspirin"]}, {"drugs": ["aspirin", "warfarin"]}, {"drugs": ["metoprolol"]}]
    tps, fps, fns = match_patient(preds, exp, acceptable=[])
    assert len(tps) == 1 and len(fps) == 1 and not fns


def test_every_expected_flag_has_evidence_in_a_chunk():
    gold = gold_chunks(data.load_expected())
    assert all(gold.values()), [k for k, v in gold.items() if not v]


def test_test_split_is_locked(monkeypatch):
    import sys
    from src import evaluate
    monkeypatch.setattr(sys, "argv", ["evaluate", "--system", "keyword", "--split", "test"])
    with pytest.raises(SystemExit):
        evaluate.main()


def _quote_around(chunk_id, phrase, extra=6):
    """A real-looking quote: the evidence phrase plus a few words before and after it, copied from the chunk."""
    words = get_chunk(chunk_id)["text"].split()
    first = phrase.split()[0].lower().strip(",.()")
    n = len(phrase.split())
    for i, w in enumerate(words):
        if w.lower().strip(",.()").startswith(first) and \
                " ".join(words[i:i + n]).lower().startswith(" ".join(phrase.split()[:2]).lower()):
            return " ".join(words[max(0, i - extra):i + n + extra])
    return phrase


@pytest.fixture()
def fake_ai(monkeypatch):
    """A fake AI that answers with the correct warnings, quoting the gold evidence chunk,
    plus one made-up warning with an invented quote (which the citation check must remove)."""
    expected = data.load_expected()
    gold = gold_chunks(expected)

    def fake_call(system, user, schema, model=None, use_cache=True):
        import re
        if schema is prompts.REPAIR_SCHEMA:     # quote repair (v3): the invented warning has no real passage
            return {"output": {"found": False, "quote": ""}, "usage": {}, "model": "fake", "cached": False}
        pid = re.search(r"\((P\d{3})\)", user).group(1)
        flags = []
        for e in expected[expected.patient_id == pid].itertuples():
            cid = gold[e.flag_id][0]
            flags.append({"flag_type": e.flag_type, "drugs": sorted(e.drug_set), "severity": e.severity,
                          "explanation": e.expected_message, "recommendation": "",
                          "citations": [{"chunk_id": cid, "quote": _quote_around(cid, e.evidence_phrase)}]})
        flags.append({"flag_type": "interaction", "drugs": ["warfarin", "metoprolol"], "severity": "major",
                      "explanation": "invented", "recommendation": "",
                      "citations": [{"chunk_id": "warfarin::drug_interactions::00",
                                     "quote": "metoprolol doubles the anticoagulant effect of warfarin in all patients"}]})
        return {"output": {"flags": flags, "not_in_library": []}, "usage": {"input_tokens": 1, "output_tokens": 1},
                "model": "fake", "cached": False}

    monkeypatch.setattr(systems, "call_json", fake_call)
    # put the patient id into the prompt so the fake AI knows which patient it is answering
    monkeypatch.setattr(systems, "patient_summary", lambda p: f"({p['patient_id']}) " + data.patient_summary(p))


def test_rag_end_to_end_with_fake_ai(fake_ai):
    retriever = Retriever("bm25")
    preds = [systems.rag_system(data.patient_record(r), retriever, k=5)
             for r in data.load_patients("dev").itertuples()]
    metrics, _ = score(preds, "dev")
    # every invented warning is removed by the citation check ...
    assert all(f["drugs"] != ["warfarin", "metoprolol"] for p in preds for f in p["flags"])
    assert any(p["removed_flags"] for p in preds)
    # ... and correct warnings whose evidence was retrieved are kept and scored as grounded
    assert metrics["recall"] >= 0.9 and metrics["precision"] == 1.0
    assert metrics["citation_accuracy"] == 1.0


# --- Free AI providers (simulated servers, no internet needed) -------------------------
import json as _json


def _fake_answer():
    return {"flags": [{"flag_type": "Interaction", "drugs": ["warfarin", "aspirin"], "severity": "Major",
                       "explanation": "bleeding", "recommendation": "monitor",
                       "citations": [{"chunk_id": "x", "quote": "q"}]}], "not_in_library": []}


def test_parse_json_handles_code_fences_and_extra_text():
    from src.llm import parse_json
    assert parse_json('```json\n{"flags": []}\n```') == {"flags": []}
    assert parse_json('Here you go: {"flags": [], "not_in_library": []} Hope this helps') == {
        "flags": [], "not_in_library": []}


def test_gemini_request_and_fallback(monkeypatch, tmp_path):
    """Gemini via its OpenAI-compatible endpoint: schema mode first, plain JSON mode if the schema is refused."""
    import httpx
    from src import config, llm
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(config, "LLM_SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setattr(llm, "_json_mode_only", [False])
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = []

    def handler(request):
        body = _json.loads(request.content)
        seen.append(body["response_format"]["type"])
        assert request.url.path.endswith("/chat/completions") and "generativelanguage" in request.url.host
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "schema keyword not supported"}})
        return httpx.Response(200, json={
            "id": "1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": _json.dumps(_fake_answer())}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    res = llm.call_json("sys", "user", {"type": "object"}, provider="gemini", model="gemini-test", http_client=client)
    assert seen == ["json_schema", "json_object"]
    assert res["output"]["flags"][0]["drugs"] == ["warfarin", "aspirin"]
    assert res["usage"]["input_tokens"] == 100
    # second identical call comes from the cache: no request
    again = llm.call_json("sys", "user", {"type": "object"}, provider="gemini", model="gemini-test", http_client=client)
    assert again["cached"] and len(seen) == 2


def test_ollama_request(monkeypatch, tmp_path):
    import requests
    from src import config, llm
    monkeypatch.setattr(config, "CACHE", tmp_path)
    sent = {}

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": _json.dumps(_fake_answer())}, "prompt_eval_count": 50, "eval_count": 10}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    res = llm.call_json("sys", "user", {"type": "object"}, provider="ollama", model="qwen2.5:7b")
    assert sent["url"].endswith("/api/chat") and sent["body"]["format"] == {"type": "object"}
    assert sent["body"]["options"]["num_ctx"] >= 16384      # long prompts must not be cut
    assert res["model"] == "ollama:qwen2.5:7b"


def test_loose_answers_are_cleaned():
    from src.systems import _flags_from
    f = _flags_from(_fake_answer())[0]
    assert f["flag_type"] == "interaction" and f["severity"] == "major"
    assert _flags_from({"flags": [{"drugs": "warfarin"}]})[0]["drugs"] == ["warfarin"]


def _ok_response(model):
    import httpx
    return httpx.Response(200, json={
        "id": "1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": _json.dumps(_fake_answer())}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


def test_gemini_waits_when_free_limit_is_hit(monkeypatch, tmp_path, capsys):
    import httpx
    from src import config, llm
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(config, "LLM_SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setattr(llm, "_json_mode_only", [False])
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    waits = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: waits.append(s))
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "17"},
                                  json={"error": {"message": "Resource exhausted: requests per minute"}})
        return _ok_response("m")

    res = llm.call_json("s", "u", {"type": "object"}, provider="gemini", model="m",
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert res["output"]["flags"] and 17 in waits and len(calls) == 2
    assert "waiting 17s" in capsys.readouterr().out          # the user can see why it is waiting
    assert config.LLM_SECONDS_BETWEEN_CALLS >= 12            # and the run slows down to avoid more waits


def test_gemini_daily_limit_stops_with_clear_message(monkeypatch, tmp_path):
    import httpx
    from src import config, llm
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(config, "LLM_SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def handler(request):
        return httpx.Response(429, json={"error": {"message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel"}})

    with pytest.raises(llm.LLMError, match="daily"):
        llm.call_json("s", "u", {"type": "object"}, provider="gemini", model="m",
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_gemini_retries_when_server_is_busy(monkeypatch, tmp_path, capsys):
    import httpx
    from src import config, llm
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(config, "LLM_SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setattr(llm, "_json_mode_only", [False])
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(503, json={"error": {"message": "The model is overloaded. Please try again later."}})
        return _ok_response("m")

    res = llm.call_json("s", "u", {"type": "object"}, provider="gemini", model="m",
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert res["output"]["flags"] and len(calls) == 3
    assert "server busy" in capsys.readouterr().out


# --- Week 3 guardrails ----------------------------------------------------------------------
def test_quote_may_skip_text_but_never_invent_it():
    text = ("Dosage regimens for patients with severe renal impairment are described in Table 1 "
            "[see Use in Specific Populations (8.7)]. Table 1: Prophylaxis in abdominal surgery 30 mg once daily. "
            "Prophylaxis in hip or knee replacement surgery 30 mg once daily.")
    skipped = ("Dosage regimens for patients with severe renal impairment are described in Table 1. "
               "Prophylaxis in hip or knee replacement surgery 30 mg once daily")
    assert quote_in_text(skipped, text)                                        # exact pieces, in order
    assert not quote_in_text("Dosage regimens for patients with mild renal impairment", text)   # changed word
    assert not quote_in_text("Prophylaxis in hip or knee replacement surgery. Dosage regimens for patients with", text)  # wrong order


def test_large_patients_are_reviewed_one_medicine_at_a_time(fake_ai):
    retriever = Retriever("bm25")
    rows = data.load_patients()
    big = data.patient_record(rows[rows.patient_id == "P033"].iloc[0])      # 5 medicines
    small = data.patient_record(rows[rows.patient_id == "P001"].iloc[0])    # 3 medicines
    v2_big = systems.rag_system(big, retriever, k=5, variant="v2")
    assert v2_big["meta"]["requests"] == len(big["in_scope_drugs"])
    assert systems.rag_system(big, retriever, k=5, variant="v1")["meta"]["requests"] == 1
    assert systems.rag_system(small, retriever, k=5, variant="v2")["meta"]["requests"] == 1
    # the same warning found from several medicines' requests is shown once
    keys = [frozenset(f["drugs"]) for f in v2_big["flags"]]
    assert len(keys) == len(set(keys))


def test_week2_prompt_is_unchanged_so_saved_answers_are_reused():
    from src import prompts
    assert prompts.RAG_SYSTEM.startswith(prompts.RAG_SYSTEM_V1) and prompts.RAG_SYSTEM != prompts.RAG_SYSTEM_V1
    assert prompts.rag_user("p", ["a"], []).endswith("Review this patient's medicines and return the flags.")


# --- Week 3 edge cases ------------------------------------------------------------------------
def test_brand_names_recognised_and_misspellings_only_suggested():
    assert data.normalize_drug("Plavix") == "clopidogrel"
    assert data.normalize_drug("Toprol XL") == "metoprolol"
    assert data.normalize_drug("warfrin") is None                 # never guessed ...
    assert data.suggest_drug("warfrin") == "warfarin"             # ... only suggested
    assert data.suggest_drug("hydroxyzine") is None               # a different drug: no risky suggestion


def test_stress_test_checks():
    import pandas as pd
    from src.stress_test import check_case
    case = pd.Series({"must_flag": "warfarin + ibuprofen", "must_not_flag": float("nan"), "expect_no_flags": "no",
                      "must_list_not_checked": float("nan"), "same_as": float("nan")})
    good = {"flags": [{"drugs": ["warfarin", "ibuprofen"], "severity": "major"}], "not_in_library": []}
    bad = {"flags": [], "not_in_library": []}
    row = lambda s: type("R", (), s.to_dict())
    assert check_case(row(case), good, {}) == (True, [])
    assert check_case(row(case), bad, {})[0] is False


def test_the_ai_cannot_decide_what_was_checked(fake_ai, monkeypatch):
    """Week 3 finding (E01): the AI listed recognised brand names as 'not checked'. Only our code decides."""
    real = systems.call_json

    def ai_with_wrong_list(*a, **k):
        res = real(*a, **k)
        res["output"]["not_in_library"] = ["Coumadin", "warfarin", "aspirin"]
        return res

    monkeypatch.setattr(systems, "call_json", ai_with_wrong_list)
    rows = data.load_patients()
    p = data.patient_record(rows[rows.patient_id == "P025"].iloc[0])     # propranolol + albuterol inhaler
    out = systems.rag_system(p, Retriever("bm25"), k=5)
    assert out["not_in_library"] == ["albuterol inhaler"]


# --- Week 4: RAG v3 (self-check and quote repair) --------------------------------------
def _one_patient():
    from types import SimpleNamespace
    return data.patient_record(SimpleNamespace(
        patient_id="T1", age=85, sex="F", weight_kg=52, serum_creatinine_mg_dl=1.0, pregnant="No",
        conditions="atrial fibrillation", allergies="none", medications="apixaban 5 mg twice daily"))


def _v3_fake(monkeypatch, first_answer, repair_answer=None):
    calls = []

    def fake_call(system, user, schema, model=None, use_cache=True):
        calls.append(schema)
        if schema is prompts.REPAIR_SCHEMA:
            return {"output": repair_answer, "usage": {"seconds": 1}, "model": "fake", "cached": False}
        return {"output": first_answer(user), "usage": {"seconds": 1}, "model": "fake", "cached": False}
    monkeypatch.setattr(systems, "call_json", fake_call)
    return calls


def _first_chunk_id(user):
    import re
    return re.search(r'<excerpt id="([^"]+)" drug="apixaban"', user).group(1)


def test_v3_self_check_drops_flags_the_ai_says_are_acceptable(monkeypatch):
    from src.retrieval import get_chunk
    def answer(user):
        cid = _first_chunk_id(user)
        quote = " ".join(get_chunk(cid)["text"].split()[:12])
        return {"flags": [{"flag_type": "dose", "drugs": ["apixaban"], "severity": "major", "is_problem": False,
                           "explanation": "The dose matches the label for this patient.", "recommendation": "",
                           "citations": [{"chunk_id": cid, "quote": quote}]}]}
    _v3_fake(monkeypatch, answer)
    out = systems.rag_system(_one_patient(), Retriever("bm25"), variant="v3")
    assert out["flags"] == [] and len(out["dismissed_flags"]) == 1
    # v2 has no self-check, so the same answer is still shown there
    out2 = systems.rag_system(_one_patient(), Retriever("bm25"), variant="v2")
    assert len(out2["flags"]) == 1


def test_v3_repairs_a_reworded_quote(monkeypatch):
    from src.retrieval import get_chunk
    holder = {}
    def answer(user):
        cid = holder.setdefault("cid", _first_chunk_id(user))
        return {"flags": [{"flag_type": "dose", "drugs": ["apixaban"], "severity": "major", "is_problem": True,
                           "explanation": "Two of age, weight, creatinine: reduce the dose.", "recommendation": "",
                           "citations": [{"chunk_id": cid, "quote": "a sentence the label never says in these words"}]}]}
    _v3_fake(monkeypatch, answer, repair_answer=None)
    # repair finds nothing -> the warning stays in the "could not be verified" list
    out = systems.rag_system(_one_patient(), Retriever("bm25"), variant="v3")
    assert out["flags"] == [] and len(out["removed_flags"]) == 1
    # repair copies a real passage -> the warning is kept and marked as repaired
    real = " ".join(get_chunk(holder["cid"])["text"].split()[5:25])
    calls = _v3_fake(monkeypatch, answer, repair_answer={"found": True, "quote": real})
    out = systems.rag_system(_one_patient(), Retriever("bm25"), variant="v3")
    assert len(out["flags"]) == 1 and out["flags"][0]["repaired"] and out["flags"][0]["citations"][0]["verified"]
    assert out["meta"]["repaired"] == 1 and calls.count(prompts.REPAIR_SCHEMA) == 1
    assert out["meta"]["requests"] == len(calls)
