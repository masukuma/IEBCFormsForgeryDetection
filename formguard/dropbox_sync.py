"""Incremental local mirror of a Dropbox folder (requires `pip install dropbox`).

Files are skipped when the local copy already has the same Dropbox content hash, so re-running
after new submissions arrive only downloads the new or changed files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BLOCK = 4 * 1024 * 1024


def content_hash(path: Path) -> str:
    """Dropbox's content hash: SHA-256 over the concatenated SHA-256 digests of 4 MiB blocks."""
    digests = b""
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            digests += hashlib.sha256(block).digest()
    return hashlib.sha256(digests).hexdigest()


def sync(token: str, remote: str, local: Path) -> int:
    import dropbox
    from dropbox.files import FileMetadata

    dbx = dropbox.Dropbox(token)
    res = dbx.files_list_folder(remote, recursive=True)
    entries = list(res.entries)
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        entries.extend(res.entries)

    downloaded = 0
    prefix = remote.rstrip("/").lower()
    for e in entries:
        if not isinstance(e, FileMetadata):
            continue
        rel = e.path_display[len(prefix) :].lstrip("/")
        dest = local / rel
        if dest.exists() and content_hash(dest) == e.content_hash:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dbx.files_download_to_file(str(dest), e.path_lower)
        downloaded += 1
    return downloaded
