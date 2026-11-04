"""Tests for the web app (app.py), with a fake AI model: no internet or API key needed.
Run:  python -m pytest tests/test_app.py
"""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import data, systems  # noqa: E402
from src.retrieval import gold_chunks, get_chunk  # noqa: E402


@pytest.fixture()
def fake_ai(monkeypatch):
    """Answers P001 correctly (warfarin + aspirin, quoting the real label) and adds one invented warning."""
    expected = data.load_expected()
    gold = gold_chunks(expected)
    f001 = expected[expected.flag_id == "F001"].iloc[0]
    words = get_chunk(gold["F001"][0])["text"].split()
    start = next(i for i, w in enumerate(words) if w == "Antiplatelet")
    real_quote = " ".join(words[start - 6:start + 8])

    def fake_call(system, user, schema, model=None, use_cache=True):
        flags = [{"flag_type": "interaction", "drugs": ["warfarin", "aspirin"], "severity": "major",
                  "explanation": "Aspirin increases bleeding risk with warfarin.", "recommendation": "Monitor INR.",
                  "citations": [{"chunk_id": gold["F001"][0], "quote": real_quote}]},
                 {"flag_type": "interaction", "drugs": ["warfarin", "metoprolol"], "severity": "major",
                  "explanation": "invented", "recommendation": "",
                  "citations": [{"chunk_id": gold["F001"][0], "quote": "metoprolol doubles the effect of warfarin in all"}]}]
        return {"output": {"flags": flags, "not_in_library": []}, "usage": {"input_tokens": 1, "output_tokens": 1},
                "model": "fake", "cached": False}

    monkeypatch.setattr(systems, "call_json", fake_call)
    return f001


def test_app_loads_with_disclaimer():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert any("not for clinical use" in w.value.lower() for w in at.warning)


def test_app_checks_a_practice_patient(fake_ai):
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    at.sidebar.selectbox[0].set_value("P001").run()        # load practice patient P001
    assert at.number_input(key="age").value == 72
    at.button[0].click().run()                              # "Check medicines"
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "warfarin + aspirin" in " ".join(e.value for e in at.warning)   # MAJOR warning shown
    assert "Verified in the official label" in text
    assert "metoprolol doubles" not in text                  # the invented quote is never shown as evidence
    # the warning whose quote failed is shown clearly as "could not be verified" (Week 4), not as a real warning
    assert any("1 possible problem(s) could not be verified" in e.value for e in at.warning)
    assert "**warfarin + metoprolol** (interaction): invented" in text
