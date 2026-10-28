"""Ollama Cloud models: a vision model reads figures from evidence crops, a text model drafts prose.

Both are optional and both are fenced in:

* Readings from the vision model are recorded with the model name and shown next to the crop
  they came from, so every figure in the report can be checked against its image.
* Drafted prose may only restate the facts it is given. Any number in a draft that is not in
  the facts rejects the draft, and the deterministic template text is used instead. Detection,
  measurements and verdicts never come from a language model.

Configuration (environment):
    OLLAMA_API_KEY            required to use the models (https://ollama.com/settings/keys)
    OLLAMA_HOST               default https://ollama.com
    FORMGUARD_VISION_MODEL    default kimi-k3
    FORMGUARD_TEXT_MODEL      default kimi-k3
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com").rstrip("/")
VISION_MODEL = os.environ.get("FORMGUARD_VISION_MODEL", "kimi-k3")
TEXT_MODEL = os.environ.get("FORMGUARD_TEXT_MODEL", "kimi-k3")


def available() -> bool:
    return bool(os.environ.get("OLLAMA_API_KEY"))


def chat(model: str, messages: list[dict], fmt: dict | str | None = None, timeout: int = 300) -> str:
    body: dict = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if fmt is not None:
        body["format"] = fmt
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"]


READ_SCHEMA = {
    "type": "object",
    "properties": {
        "original": {"type": "string", "description": "figures in the left (original) panel, top to bottom, comma-separated"},
        "questioned": {"type": "string", "description": "figures in the right (questioned) panel, top to bottom, comma-separated"},
        "legible": {"type": "boolean"},
    },
    "required": ["original", "questioned", "legible"],
}


def read_figures(tile_png: bytes, model: str = VISION_MODEL) -> dict:
    """Read the handwritten figures in an evidence tile (original | questioned, side by side)."""
    prompt = (
        "This image shows the same area of an election results form twice: the ORIGINAL on the "
        "left and a QUESTIONED copy on the right. Transcribe the handwritten numbers in each "
        "panel exactly as written, including leading zeros, top to bottom. Do not correct or "
        "infer anything; if a number is not legible, set legible to false."
    )
    content = chat(
        model,
        [{"role": "user", "content": prompt, "images": [base64.b64encode(tile_png).decode()]}],
        fmt=READ_SCHEMA,
    )
    out = json.loads(content)
    out["model"] = model
    return out


DRAFT_SYSTEM = (
    "You draft paragraphs for an expert's affidavit in a Kenyan election petition. Write in the "
    "first person, formal, plain and neutral. Use ONLY the facts provided in the JSON. Do not add "
    "facts, numbers, dates, names, opinions about anyone's motives or conclusions that the facts do "
    "not state. Do not use headings or bullet points. Return only the paragraph text."
)


def _numbers(text: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}


def draft(instruction: str, facts: dict, fallback: str, model: str = TEXT_MODEL) -> tuple[str, str]:
    """Draft one paragraph from facts. Returns (text, source) where source names the model or 'template'."""
    if not available():
        return fallback, "template"
    try:
        text = chat(
            model,
            [
                {"role": "system", "content": DRAFT_SYSTEM},
                {"role": "user", "content": f"{instruction}\n\nFACTS:\n{json.dumps(facts, indent=2, default=str)}"},
            ],
        ).strip()
    except Exception:
        return fallback, "template"
    # Every number in the draft must already appear in the facts (or the fallback built from them).
    allowed = _numbers(json.dumps(facts, default=str)) | _numbers(fallback)
    if not text or _numbers(text) - allowed:
        return fallback, "template"
    return text, model
