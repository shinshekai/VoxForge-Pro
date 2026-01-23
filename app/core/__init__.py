# VoxForge Pro Core Modules
"""
Core modules for PDF processing, TTS generation, and library management.

ENVIRONMENT VARIABLE FOR PADDLEOCR:
Setting DISABLE_MODEL_SOURCE_CHECK=True disables the connectivity check.
This must be set before importing paddleocr.
"""

import os

# Set environment variable early for paddleocr
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

__version__ = "1.0.0"
