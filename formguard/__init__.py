"""formguard: detect alterations in submitted copies of scanned forms by comparing them to the originals."""
from .pipeline import Config, compare_documents

__all__ = ["Config", "compare_documents"]
