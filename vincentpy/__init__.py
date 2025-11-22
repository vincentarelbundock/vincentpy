"""
vincentpy package initialization.

Expose the DuckDB and LangChain helper modules at the package level so callers
can import them via ``import vincentpy.duckdb`` or ``from vincentpy import duckdb``.
"""

from . import duckdb, langchain, utils

__all__ = ["duckdb", "langchain", "utils"]
