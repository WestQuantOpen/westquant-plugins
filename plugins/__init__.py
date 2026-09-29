"""WestQuant public plugins package.

Importing this package does not require any framework dependencies.
Adapters are imported lazily when accessed.
"""
from .westquant_sdk.sdk import Search

__all__ = ["Search"]


def __getattr__(name):
    """Lazy-load adapters so missing framework deps don't break import."""
    if name == "QiskitAdapter":
        from .qiskit.adapter import QiskitAdapter
        return QiskitAdapter
    if name == "TKETAdapter":
        from .tket.adapter import TKETAdapter
        return TKETAdapter
    if name == "PyZXAdapter":
        from .pyzx.adapter import PyZXAdapter
        return PyZXAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
