"""Integrity facts about a reference file: hash, and the embedded PDF signature if it has one.

The 2022 Forms 34A on IEBC's portal carry a detached PKCS#7 signature from the KIEMS kit
(Smartmatic certificate). Validating it shows whether the file is byte-for-byte what was signed
at upload. pyHanko is optional; without it the signature is reported as present but unchecked.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .render import parse_source


def file_facts(source: str | Path) -> dict:
    path, pages = parse_source(source)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out = {"file": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    if pages:
        out["pages"] = pages
    return out


def pdf_signatures(source: str | Path) -> list[dict]:
    path, _ = parse_source(source)
    if path.suffix.lower() != ".pdf":
        return []
    try:
        from pyhanko.pdf_utils.reader import PdfFileReader
        from pyhanko.sign.validation import validate_pdf_signature
        from pyhanko_certvalidator import ValidationContext
    except ImportError:
        data = path.read_bytes()
        return [{"present": True, "checked": False, "reason": "pyhanko not installed"}] if b"/Type /Sig" in data else []

    # pyHanko logs a full traceback for the KIEMS kits' SHA-1 digest (deprecated by its policy);
    # the signature is still validated and the algorithm is reported in the result.
    logging.getLogger("pyhanko").setLevel(logging.CRITICAL)
    out = []
    with open(path, "rb") as fh:
        reader = PdfFileReader(fh, strict=False)
        for sig in reader.embedded_signatures:
            entry: dict = {"present": True, "checked": True, "field": str(sig.field_name)}
            try:
                st = validate_pdf_signature(sig, ValidationContext(allow_fetching=False))
                entry.update(
                    intact=bool(st.intact),  # signed bytes unchanged
                    valid=bool(st.valid),  # signature verifies against the signer's key
                    coverage=st.coverage.name,  # ENTIRE_FILE = nothing appended after signing
                    modification_level=st.modification_level.name if st.modification_level else None,
                    trusted=bool(st.trusted),  # chain to a trusted root (none configured here)
                    digest_algorithm=str(st.md_algorithm),
                    signer=str(st.signing_cert.subject.human_friendly),
                    issuer=str(st.signing_cert.issuer.human_friendly),
                    signing_time=st.signer_reported_dt.isoformat() if st.signer_reported_dt else None,
                )
            except Exception as e:  # report, never hide, a signature that cannot be checked
                entry.update(checked=False, error=f"{type(e).__name__}: {e}")
            out.append(entry)
    return out
