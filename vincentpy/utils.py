"""
General utility helpers shared across vincentpy modules.
"""

from __future__ import annotations

import os


def get_env_var(name: str) -> str:
    """Return an environment variable and raise a helpful error if missing."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Please set the {name} environment variable.")
    return value


__all__ = ["get_env_var"]
