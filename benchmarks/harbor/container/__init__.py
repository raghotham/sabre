"""
SABRE Agent for Harbor benchmarks.

This package provides a Harbor-compatible agent that communicates with
a separately-running SABRE server via HTTP.
"""

from .agent import SabreAgent

__all__ = ["SabreAgent"]
