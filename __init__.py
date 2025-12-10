"""
Caption Live - ComfyUI Custom Node
C++ powered caption effects.
"""
import os
import sys

# Add current directory for engine import
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .caption_live import CaptionLiveNode

NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live",
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("✅ Caption Live Loaded!")