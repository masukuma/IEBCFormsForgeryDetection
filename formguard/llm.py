"""Ollama Cloud models: a vision model reads figures from evidence crops, a text model drafts prose.

Cheapest model first. Each task walks a *ladder* of models from the smallest (cheapest, and on
Ollama Cloud's free tier the least quota per call) to the most capable, and moves up a rung only
when the cheaper model's output fails validation. Most calls end on the first rung; the usage log
records which model answered each call and its token counts.

Both uses are fenced in:

* A reading must be valid JSON of digits, marked legible, or the next rung is tried. Readings are
  recorded with the model that made them and shown next to the crop they came from, so every
  figure in the report can be checked against its image. ``confirm=True`` also asks the next rung
  for an independent second reading and records whether the two agree.
* Drafted prose may only restate the facts it is given: any number not in the facts, or a draft
  that does not open with "THAT", fails validation. If every rung fails, the deterministic template
  text is used. Detection, measurements and verdicts never come from a language model.

Configuration (environment):
    OLLAMA_API_KEY            required to use the models
    OLLAMA_HOST               default https://ollama.com
    FORMGUARD_VISION_MODELS   comma-separated ladder, cheapest first
    FORMGUARD_TEXT_MODELS     comma-separated ladder, cheapest first
"""
from __future__ import annotations

import base64
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com").rstrip("/")
# Smallest first. gemma4:31b is the smallest vision model on Ollama Cloud; gpt-oss:20b the smallest text model.
VISION_LADDER = os.environ.get("FORMGUARD_VISION_MODELS", "gemma4:31b,kimi-k2.6,kimi-k3").split(",")
TEXT_LADDER = os.environ.get("FORMGUARD_TEXT_MODELS", "gpt-oss:20b,nemotron-3-nano:30b,gpt-oss:120b,kimi-k3").split(",")

USAGE: list[dict] = []  # one entry per call: model, task, ok, tokens, seconds


def available() -> bool:
    return bool(os.environ.get("OLLAMA_API_KEY"))


def chat(model: str, messages: list[dict], fmt: dict | str | None = None, timeout: int = 300, task: str = "") -> str:
    body: dict = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if fmt is not None:
        body["format"] = fmt
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    USAGE.append(
        {
            "model": model,
            "task": task,
            "prompt_tokens": out.get("prompt_eval_count"),
            "output_tokens": out.get("eval_count"),
            "seconds": round(time.time() - t0, 1),
        }
    )
    return out["message"]["content"]


def usage_summary() -> dict:
    by: dict[str, dict] = {}
    for u in USAGE:
        s = by.setdefault(u["model"], {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "accepted": 0})
        s["calls"] += 1
        s["prompt_tokens"] += u.get("prompt_tokens") or 0
        s["output_tokens"] += u.get("output_tokens") or 0
        s["accepted"] += 1 if u.get("ok") else 0
    return by


def _ladder(models: list[str], task: str, call, valid) -> tuple[object | None, str | None, list[str]]:
    """Try each model in turn; return (first valid result, model, failures)."""
    failures = []
    for model in models:
        try:
            result = call(model)
        except (OSError, http.client.HTTPException, ValueError, KeyError) as e:  # network, HTTP, bad JSON
            failures.append(f"{model}: {type(e).__name__}")
            continue
        ok = valid(result)
        if USAGE:
            USAGE[-1]["ok"] = ok
        if ok:
            return result, model, failures
        failures.append(f"{model}: failed validation")
    return None, None, failures


# ---------------------------------------------------------------------------------------------
# readings

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "original": {"type": "string", "description": "numbers in the LEFT panel, top to bottom, comma-separated"},
        "questioned": {"type": "string", "description": "numbers in the RIGHT panel, top to bottom, comma-separated"},
        "legible": {"type": "boolean"},
    },
    "required": ["original", "questioned", "legible"],
}
READ_PROMPT = (
    "This image shows the same small area of an election results form twice: the ORIGINAL on the "
    "left (under the word 'original') and a QUESTIONED copy on the right (under 'submission'). "
    "Transcribe the handwritten numbers in each panel exactly as written, including leading zeros, "
    "top to bottom, comma-separated. Ignore printed words and ruled lines. Do not correct, infer or "
    "complete anything. If any number is not clearly legible, set legible to false."
)


def _unfence(text: str) -> str:
    """Some models wrap JSON in a ```json fence even when a format is requested."""
    m = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return m.group(1) if m else text


def _valid_reading(r) -> bool:
    if not isinstance(r, dict) or not r.get("legible"):
        return False
    pat = re.compile(r"^\s*\d+(\s*,\s*\d+)*\s*$")
    return all(isinstance(r.get(k), str) and pat.match(r[k]) for k in ("original", "questioned"))


def read_figures(tile_png: bytes, confirm: bool = False) -> dict | None:
    """Read the handwritten figures in an evidence tile, cheapest vision model first."""
    img = base64.b64encode(tile_png).decode()

    def call(model):
        return json.loads(_unfence(chat(model, [{"role": "user", "content": READ_PROMPT, "images": [img]}], fmt=READ_SCHEMA, task="read")))

    result, model, failures = _ladder(VISION_LADDER, "read", call, _valid_reading)
    if result is None:
        return {"legible": False, "failures": failures}
    norm = lambda s: ",".join(x.strip() for x in s.split(","))
    out = {"original": norm(result["original"]), "questioned": norm(result["questioned"]), "by": model, "failures": failures}
    if confirm:
        rest = VISION_LADDER[VISION_LADDER.index(model) + 1 :]
        second, model2, _ = _ladder(rest, "confirm", call, _valid_reading)
        if second is not None:
            agree = norm(second["original"]) == out["original"] and norm(second["questioned"]) == out["questioned"]
            out["confirmed_by"] = model2
            out["confirmed"] = agree
            if not agree:
                out["second_reading"] = {"original": norm(second["original"]), "questioned": norm(second["questioned"])}
    return out


# ---------------------------------------------------------------------------------------------
# drafting

DRAFT_SYSTEM = (
    "You draft paragraphs for an expert's affidavit in a Kenyan election petition. Write in the "
    "first person, in the plain register of a sworn affidavit: short declarative sentences, past "
    "tense for what was done, no rhetoric. Use ONLY the facts provided in the JSON. Do not add "
    "facts, numbers, dates, names, opinions about anyone's motives or conclusions that the facts do "
    "not state. No headings, lists, emphasis or dashes. No intensifiers or stock transitions such as "
    "'clearly', 'notably', 'crucially', 'furthermore' or 'moreover'. Do not announce what the "
    "paragraph will say or sum it up. Begin with the word THAT. Return only the paragraph."
)

# Marks of machine-written prose that have no place in a sworn affidavit; a draft containing any
# of them is rejected and the next model, or the plain template, is used instead.
STYLE = re.compile(
    r"\u2014|\*\*|^#|^\s*[-*\u2022] |!|\b(?:crucial(?:ly)?|notably|importantly|it is (?:important|worth|essential)"
    r"|it should be noted|comprehensive|robust|furthermore|moreover|additionally|in conclusion|in summary"
    r"|delve|underscore[sd]?|highlight(?:s|ed|ing)?|pivotal|meticulous(?:ly)?|seamless(?:ly)?|leverag\w*|landscape"
    r"|testament|showcas\w*|vital|clearly|undeniabl\w*|unequivocal\w*|irrefutabl\w*|conclusively|beyond (?:any )?doubt"
    r"|striking(?:ly)?|glaring|blatant(?:ly)?|egregious)\b",
    re.IGNORECASE | re.MULTILINE,
)


def plain(text: str) -> bool:
    """True when a draft has none of the STYLE marks."""
    return not STYLE.search(text)


def _numbers(text: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}


def draft(instruction: str, facts: dict, fallback: str) -> tuple[str, str]:
    """Draft one paragraph from facts, cheapest text model first. Returns (text, model or 'template')."""
    if not available():
        return fallback, "template"
    allowed = _numbers(json.dumps(facts, default=str)) | _numbers(fallback)
    messages = [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": f"{instruction}\n\nFACTS:\n{json.dumps(facts, indent=2, default=str)}"},
    ]

    def call(model):
        text = chat(model, messages, task="draft").strip()
        return re.sub(r"^\W*\*?\*?THAT\*?\*?", "THAT", text)  # tolerate markdown bold on the opening word

    def valid(text) -> bool:
        return (
            bool(text)
            and text.startswith("THAT")
            and 40 <= len(text) <= 2 * len(fallback) + 200
            and not (_numbers(text) - allowed)
            and plain(text)
        )

    text, model, _ = _ladder(TEXT_LADDER, "draft", call, valid)
    return (text, model) if text else (fallback, "template")
