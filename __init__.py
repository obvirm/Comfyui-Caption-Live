from .caption_live import CaptionLiveNode

NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live"
}

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]