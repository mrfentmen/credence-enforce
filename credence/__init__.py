"""
Credence — epistemic enforcement layer for AI-assisted coding.

Tracks uncertain values from conversation and blocks writes until verified.
Works with any coding agent. Zero API key required.

Primary interface — MCP server:
    pip install credence-enforce
    credence-server              # starts the MCP server

Zero-API Python interface:
    CredenceRegistry             # SQLite constraint store
    CredenceMemory               # cross-session epistemic memory
    wrap(fn, context)            # wrap any compress function with faithfulness guard

CLI:
    credence demo                # 30-second smoke test, no setup
    credence stats               # false-positive rate from real usage
    credence feedback 1|2|3      # tag last gate block as TP/FP/skip
"""

from .registry import CredenceRegistry
from .memory import CredenceMemory
from .wrap import wrap, WrapResult, measure_fcr

__version__ = "1.3.0"
__all__ = [
    "CredenceRegistry",
    "CredenceMemory",
    "wrap", "WrapResult", "measure_fcr",
]
