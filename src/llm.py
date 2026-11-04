"""Talking to the AI model, with structured JSON output. Free options first.

Providers (set LLM_PROVIDER in .env):
  gemini     Google Gemini free tier (default). Free API key, no credit card: https://aistudio.google.com/apikey
  ollama     A free open model running on your own laptop (https://ollama.com). No account, fully offline.
  anthropic  Claude (paid). Optional.

Every answer is saved in results/cache/ (keyed by provider + model + prompt), so:
  * re-running the evaluation makes no new requests and gives the same answer (reproducible results)
  * judges can inspect exactly what the model returned
"""
import hashlib
import json
import os
import re
import time

from . import config


class LLMError(RuntimeError):
    pass


def _cache_path(payload: dict):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return config.CACHE / f"{key}.json"


def default_model(provider):
    return {"gemini": config.GEMINI_MODEL, "ollama": config.OLLAMA_MODEL,
            "anthropic": config.CLAUDE_MODEL}[provider]


def parse_json(text: str) -> dict:
    """Read the model's JSON even if it is wrapped in ```json fences or has text around it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"The model did not return JSON: {text[:200]!r}")
        return json.loads(text[start:end + 1])


def _schema_instruction(schema):
    return ("\n\nReturn ONLY one JSON object (no other text) that follows this JSON schema:\n"
            + json.dumps(schema, separators=(",", ":")))


# --- Providers ---------------------------------------------------------------------------
_last_call = [0.0]


def _pace():
    """Stay under free-tier rate limits (e.g. ~10 requests per minute)."""
    wait = config.LLM_SECONDS_BETWEEN_CALLS - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


_json_mode_only = [False]   # remembered after the first refusal, so later calls don't waste a request


def _retry_delay(err, default):
    """Seconds Google asks us to wait (from the 'retry-after' header or the error body)."""
    try:
        header = err.response.headers.get("retry-after")
        if header:
            return float(header)
    except Exception:
        pass
    m = re.search(r'retry(?:Delay| in)["\':\s]*([0-9.]+)\s*s', str(err), re.I)
    return float(m.group(1)) if m else default


def _gemini(system, user, schema, model, http_client=None):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                       "and put it in your .env file.")
    from openai import (APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI,
                        RateLimitError)
    # Retries are handled below (not hidden inside the library), so you can see when the free tier makes us wait.
    client = OpenAI(api_key=key, base_url=config.GEMINI_BASE_URL, max_retries=0,
                    timeout=config.LLM_TIMEOUT_SECONDS, http_client=http_client)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user + _schema_instruction(schema)}]
    schema_format = {"type": "json_schema", "json_schema": {"name": "flags", "schema": schema}}

    for attempt in range(1, 9):
        _pace()
        try:
            if not _json_mode_only[0]:
                try:   # 1st choice: the model is forced to follow the schema
                    resp = client.chat.completions.create(model=model, messages=messages, temperature=0,
                                                          response_format=schema_format)
                except BadRequestError as e:
                    if "model" in str(e).lower() and "not" in str(e).lower() and "found" in str(e).lower():
                        raise LLMError(f"Gemini does not know the model '{model}'. List yours with: python -m src.llm")
                    _json_mode_only[0] = True    # this model doesn't accept the schema: use plain JSON mode
                    _pace()
            if _json_mode_only[0]:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=0,
                                                      response_format={"type": "json_object"})
            usage = resp.usage
            return (resp.choices[0].message.content,
                    {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                     "output_tokens": getattr(usage, "completion_tokens", 0) or 0}, model)
        except RateLimitError as e:
            text = str(e)
            if re.search(r"per ?day|PerDay|daily", text, re.I):
                raise LLMError("The free daily limit for this model is used up. Finished patients are saved. "
                               "Run the same command again tomorrow, or try another free model "
                               "(list them: python -m src.llm --check).")
            wait = min(_retry_delay(e, 20 * attempt), 120)
            # slow down for the rest of the run, so we stop hitting the limit
            config.LLM_SECONDS_BETWEEN_CALLS = min(max(config.LLM_SECONDS_BETWEEN_CALLS * 1.5, 12), 60)
            print(f"\n    free-tier limit reached: waiting {wait:.0f}s (try {attempt}/8); "
                  f"now {config.LLM_SECONDS_BETWEEN_CALLS:.0f}s between requests", flush=True)
            time.sleep(wait)
        except InternalServerError as e:      # 500/503: Google's free servers are busy for a moment
            wait = min(10 * attempt, 60)
            print(f"\n    Gemini server busy ({getattr(e, 'status_code', 'error')}): waiting {wait}s "
                  f"(try {attempt}/8)", flush=True)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError) as e:
            print(f"\n    no answer from Gemini ({type(e).__name__}); retrying (try {attempt}/8)", flush=True)
            time.sleep(5 * attempt)
    raise LLMError("Gemini kept failing (busy or rate-limited). Wait a few minutes and run the same command "
                   "again; finished patients are saved and will not be asked again.")


def _ollama(system, user, schema, model):
    import requests
    body = {"model": model, "stream": False, "format": schema,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user + _schema_instruction(schema)}],
            # num_ctx: our prompts are ~10,000 tokens; Ollama's default context is smaller and would cut them
            "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX}}
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=1800)
    except requests.ConnectionError:
        raise LLMError(f"Cannot reach Ollama at {config.OLLAMA_URL}. Install it from https://ollama.com, "
                       f"then run:  ollama pull {model}")
    if r.status_code == 404:
        raise LLMError(f"Ollama model '{model}' not found. Run:  ollama pull {model}")
    r.raise_for_status()
    data = r.json()
    return (data["message"]["content"],
            {"input_tokens": data.get("prompt_eval_count", 0), "output_tokens": data.get("eval_count", 0)}, model)


def _anthropic(system, user, schema, model):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise LLMError("ANTHROPIC_API_KEY is not set in your .env file.")
    import anthropic
    resp = anthropic.Anthropic(max_retries=4).messages.create(
        model=model, max_tokens=config.MAX_OUTPUT_TOKENS, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}})
    if resp.stop_reason == "max_tokens":
        raise LLMError("The answer was cut off. Increase MAX_OUTPUT_TOKENS in src/config.py.")
    text = next(b.text for b in resp.content if b.type == "text")
    return text, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}, resp.model


def call_json(system: str, user: str, schema: dict, model: str = None, provider: str = None,
              use_cache=True, http_client=None) -> dict:
    """Send one request; return {"output": <parsed JSON>, "usage": {...}, "model": ..., "cached": bool}."""
    provider = provider or config.LLM_PROVIDER
    model = model or default_model(provider)
    payload = {"provider": provider, "model": model, "system": system, "user": user, "schema": schema}
    path = _cache_path(payload)
    if use_cache and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return {"output": saved["output"], "usage": saved["usage"], "model": saved["model"], "cached": True}

    t0 = time.time()
    if provider == "gemini":
        text, usage, used_model = _gemini(system, user, schema, model, http_client)
    elif provider == "ollama":
        text, usage, used_model = _ollama(system, user, schema, model)
    elif provider == "anthropic":
        text, usage, used_model = _anthropic(system, user, schema, model)
    else:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}' (use gemini, ollama or anthropic)")
    usage["seconds"] = round(time.time() - t0, 1)
    result = {"output": parse_json(text), "usage": usage, "model": f"{provider}:{used_model}"}
    config.CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**result, "raw_text": text, "request": payload}, indent=1), encoding="utf-8")
    return {**result, "cached": False}


def list_models(check=False):
    """python -m src.llm           -> the model names your Gemini key can use
       python -m src.llm --check   -> also sends one tiny request to each 'flash' model and times it"""
    from openai import OpenAI, RateLimitError
    client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url=config.GEMINI_BASE_URL, max_retries=0, timeout=60)
    names = sorted(m.id.replace("models/", "") for m in client.models.list())
    for name in names:
        if not check:
            print(name)
            continue
        if "flash" not in name or any(x in name for x in ("image", "tts", "audio", "live", "embedding")):
            continue
        t0 = time.time()
        try:
            client.chat.completions.create(model=name, messages=[{"role": "user", "content": "Reply with: ok"}])
            print(f"OK      {time.time() - t0:5.1f}s  {name}")
        except RateLimitError:
            print(f"LIMIT          {name}  (free limit reached or no free quota)")
        except Exception as e:
            print(f"ERROR          {name}  ({type(e).__name__})")
        time.sleep(4)
    if check:
        print("\nPick a fast 'OK' model and put it in .env, e.g.  GEMINI_MODEL=<name>")


if __name__ == "__main__":
    import sys
    list_models(check="--check" in sys.argv)
