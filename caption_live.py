import torch
import numpy as np
import json
import ast
import re

import os
import sys

rust_caption = None

class CaptionLiveNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "segments": ("STRING", {
                    "multiline": True, 
                    "default": '[{"start": 0.0, "end": 1.0, "text": "Hello"},{"start": 1.0, "end": 2.0, "text": "World"}]'
                }),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "font_size": ("FLOAT", {"default": 56.0, "min": 10.0, "max": 200.0}),
                "highlight_color": ("STRING", {"default": "#39E55F"}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "font_path": ("STRING", {"default": "arial.ttf"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "Custom"], {"default": "16:9"}),
                "pos_x": ("INT", {"default": 0, "min": -2000, "max": 2000}),
                "pos_y": ("INT", {"default": 0, "min": -2000, "max": 2000}),
                "style": (["box", "colored", "scaling"], {"default": "box"}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "caption-live"

    def render(self, images, segments, fps, font_size, highlight_color, text_color, font_path, aspect_ratio, pos_x, pos_y, style, mask_optional=None):
        global rust_caption
        
        # Lazy Import
        if rust_caption is None:
            try:
                # Ensure current dir is in path for local import
                current_dir = os.path.dirname(__file__)
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                    
                import rust_caption_v2 as rc
                rust_caption = rc
                print("CaptionLive: Rust backend loaded successfully.")
            except ImportError as e:
                print(f"CaptionLive CRITICAL ERROR: Failed to import 'rust_caption_v2'. Details: {e}")
                import platform
                print(f"Debug Info: Python {sys.version}, Platform {platform.platform()}")
                return (images,)
        
        # Resolve Font Path
        if not os.path.exists(font_path):
            node_dir = os.path.dirname(__file__)
            local_font = os.path.join(node_dir, font_path)
            if os.path.exists(local_font):
                font_path = local_font
            elif os.name == 'nt':
                win_font = os.path.join("C:\\Windows\\Fonts", font_path)
                if os.path.exists(win_font):
                    font_path = win_font
        
        # Parse Segments
        try:
            fixed_segments = segments.replace("'", '"')
            fixed_segments = re.sub(r',(\s*[}\]])', r'\1', fixed_segments)
            segments_data = json.loads(fixed_segments)
            if isinstance(segments_data, dict) and "segments" in segments_data:
                segments_json = json.dumps(segments_data["segments"])
            elif isinstance(segments_data, list):
                segments_json = fixed_segments
            else:
                print("CaptionLive Warning: Invalid JSON structure")
                return (images,)
        except Exception as e:
            print(f"CaptionLive JSON Parse Error: {e}")
            return (images,)
        
        # Clone output
        output_images = images.clone()
        width = int(images.shape[2])
        height = int(images.shape[1])
        
        # Auto-Scale Logic (Match visual proportion across resolutions)
        # Baseline: 512px width (common preview size approximation)
        baseline_width = 512.0
        scale_factor = width / baseline_width
        
        scaled_font_size = float(font_size) * scale_factor
        scaled_pos_x = float(pos_x) * scale_factor
        scaled_pos_y = float(pos_y) * scale_factor
        
        # Pre-calculate timestamps
        for i in range(images.shape[0]):
            try:
                time = i / fps
                
                # Returns Vec<u8> (RGBA bytes)
                rgba_bytes = rust_caption.render_mask(
                    segments_json,
                    width,
                    height,
                    time,
                    fps,
                    str(font_path),
                    style,
                    str(highlight_color),
                    str(text_color),
                    scaled_pos_x,
                    scaled_pos_y,
                    scaled_font_size
                )
                
                # Convert list/bytes to tensor (H, W, 4)
                if isinstance(rgba_bytes, list):
                    rgba_np = np.array(rgba_bytes, dtype=np.uint8)
                else:
                    rgba_np = np.frombuffer(rgba_bytes, dtype=np.uint8)
                    
                rgba_tensor = torch.from_numpy(rgba_np.reshape(height, width, 4))
                rgba_tensor = rgba_tensor.to(output_images.device)
                
                # Alpha Blending
                alpha = rgba_tensor[:, :, 3:4].float() / 255.0
                overlay_rgb = rgba_tensor[:, :, 0:3].float() / 255.0
                background_rgb = output_images[i]
                
                blended = overlay_rgb * alpha + background_rgb * (1.0 - alpha)
                output_images[i] = blended
            except Exception as e:
                if i == 0: print(f"Rust Render Error: {e}")
        
        return (output_images,)

# Node Registration
NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live (Rust)"
}
