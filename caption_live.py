import torch
import numpy as np
import json
import ast
import re

try:
    import rust_caption
except ImportError:
    print("CaptionLive Error: 'rust_caption' module not found. Please compile the Rust extension.")
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
                "style": (["box", "colored", "scaling"], {"default": "box"}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "caption-live"

    def render(self, images, segments, fps, font_size, highlight_color, text_color, font_path, style, mask_optional=None):
        if rust_caption is None:
            print("CaptionLive Error: Rust backend not loaded. Returning original images.")
            return (images,)
        
        # Parse Segments
        try:
            # Fix common JSON errors from LLM output
            fixed_segments = segments.replace("'", '"')
            # Remove trailing commas in arrays/objects
            fixed_segments = re.sub(r',(\s*[}\]])', r'\1', fixed_segments)
            
            segments_data = json.loads(fixed_segments)
            
            # Handle Whisper format {"segments": [...]}
            if isinstance(segments_data, dict) and "segments" in segments_data:
                segments_json = json.dumps(segments_data["segments"])
            # Handle list format
            elif isinstance(segments_data, list):
                segments_json = fixed_segments
            else:
                print("CaptionLive Warning: Invalid JSON structure (must be list or dict with 'segments')")
                return (images,)
        except Exception as e:
            print(f"CaptionLive JSON Parse Error: {e}")
            return (images,)
        
        # Clone output
        output_images = images.clone()
        
        # Pre-calculate timestamps for the batch
        for i in range(images.shape[0]):
            try:
                time = i / fps
                
                # Returns Vec<u8> (RGBA bytes)
                rgba_bytes = rust_caption.render_mask(
                    segments_json,
                    int(images.shape[2]),  # width
                    int(images.shape[1]),  # height
                    time,
                    style,
                    str(font_path),
                    float(font_size),
                    str(text_color),
                    str(highlight_color)
                )
                
                # Convert bytes to tensor (H, W, 4)
                rgba_tensor = torch.from_numpy(np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(images.shape[1], images.shape[2], 4))
                rgba_tensor = rgba_tensor.to(output_images.device)
                
                # Alpha Blending
                alpha = rgba_tensor[:, :, 3:4].float() / 255.0  # (H, W, 1)
                overlay_rgb = rgba_tensor[:, :, 0:3].float() / 255.0  # (H, W, 3)
                
                # Dest: output_images[i] (Background)
                background_rgb = output_images[i]  # (H, W, 3)
                
                blended = overlay_rgb * alpha + background_rgb * (1.0 - alpha)
                output_images[i] = blended
            except Exception as e:
                if i == 0: print(f"Rust Render Error: {e}")
                # Continue loop to at least return original images for other frames
        
        return (output_images,)

# Node Registration
NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live (Rust)"
}
