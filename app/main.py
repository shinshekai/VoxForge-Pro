"""
VoxForge Pro - Premium AI-Powered Audiobook Generator
CLI Entry Point

PROPER WARNING RESOLUTION (not suppression):

1. "Checking connectivity to model hosters" 
   - FIXED: Set DISABLE_MODEL_SOURCE_CHECK=True before imports

2. "pkg_resources is deprecated" 
   - CANNOT FIX: This is inside paddleocr's dependency 'perth'
   - Upstream needs to update to importlib.metadata

3. "dropout option adds dropout after all but last recurrent layer"
   - CANNOT FIX: This is in Kokoro's model architecture (harmless warning)

4. "torch.nn.utils.weight_norm is deprecated"  
   - CANNOT FIX: This is in Kokoro's model code
   - Upstream needs to migrate to torch.nn.utils.parametrizations.weight_norm

5. "Trying to convert audio from float32 to 16-bit int"
   - FIXED: Convert audio to int16 before passing to Gradio
"""

import os
import sys

# FIX #1: Set environment variable BEFORE any paddleocr-related imports
# This properly disables the connectivity check rather than suppressing the message
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from pathlib import Path

# Ensure app directory is in path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

from ui.app import main

if __name__ == "__main__":
    main()
