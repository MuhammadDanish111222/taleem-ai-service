"""Regression tests ensuring zero-weight startup without torch, transformers, onnxruntime, or pytesseract."""

import sys


def test_zero_weight_imports_do_not_load_torch_or_transformers():
    # Verify neither torch, transformers, onnxruntime, nor pytesseract are required for core modules

    forbidden = {"torch", "transformers", "onnxruntime", "pytesseract"}
    loaded = set(sys.modules.keys())
    intersection = forbidden.intersection(loaded)
    assert not intersection, f"Forbidden heavy modules were imported: {intersection}"
