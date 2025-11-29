import os
import sys
import traceback

log_file = os.path.join(os.path.dirname(__file__), "init_error.log")

def log_init(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

try:
    from .caption_live import CaptionLiveNode
    
    NODE_CLASS_MAPPINGS = {
        "CaptionLiveNode": CaptionLiveNode
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "CaptionLiveNode": "Caption Live"
    }

    WEB_DIRECTORY = "js"

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
    
    # Log success (optional, but good for confirmation)
    # log_init("Successfully loaded CaptionLiveNode")

except Exception as e:
    log_init(f"FAILED to load CaptionLiveNode: {e}")
    log_init(traceback.format_exc())
    
    # Re-raise so ComfyUI knows it failed, but we have the log now
    raise e