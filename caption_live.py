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
                "font_size": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0}),
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
        
        log_path = os.path.join(os.path.dirname(__file__), "debug.log")
        def log(msg):
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{msg}\n")
            except:
                print(msg)

        log(f"--- New Render Call ---")
        log(f"Image Shape: {images.shape}")
        
        # Lazy Import
        if rust_caption is None:
            try:
                current_dir = os.path.dirname(__file__)
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                    
                import rust_caption as rc
                rust_caption = rc
                log(f"CaptionLive: Rust backend loaded successfully from {rust_caption.__file__}.")
            except ImportError as e:
                err_msg = f"CaptionLive CRITICAL ERROR: Failed to import 'rust_caption'. Details: {e}"
                log(err_msg)
                print(err_msg)
                return (images,)
        
        # Resolve Font Path
        original_font_path = font_path
        if not os.path.exists(font_path):
            node_dir = os.path.dirname(__file__)
            local_font = os.path.join(node_dir, font_path)
            if os.path.exists(local_font):
                font_path = local_font
            elif os.name == 'nt':
                win_font = os.path.join("C:\\Windows\\Fonts", font_path)
                if os.path.exists(win_font):
                    font_path = win_font
        
        log(f"Font Path: {font_path} (Exists: {os.path.exists(font_path)})")
        log(f"Input Segments (raw): {original_font_path}") # This was a typo, should be segments

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
                log("CaptionLive Warning: Invalid JSON structure. Returning original images.")
                return (images,)
        except Exception as e:
            log(f"CaptionLive JSON Parse Error: {e}. Returning original images.")
            return (images,)
        
        log(f"Segments JSON sent to Rust: {segments_json}")
        # Clone output
        output_images = images.clone()
        width = int(images.shape[2])
        height = int(images.shape[1])
        
        baseline_width = 210.0
        scale_factor = width / baseline_width
        
        scaled_font_size = float(font_size) * scale_factor
        scaled_pos_x = float(pos_x) * scale_factor
        scaled_pos_y = float(pos_y) * scale_factor
        
        log(f"Rendering Params: W={width}, H={height}, Scale={scale_factor:.2f}, FontSize={scaled_font_size:.1f}")
        
        # Pre-calculate timestamps
        success_count = 0
        error_count = 0
        
        for i in range(images.shape[0]):
            try:
                time = i / fps
                
                # Returns Vec<u8> (RGBA bytes)
                rgba_bytes = rust_caption.render_mask_v2(
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
                
                if i == 0:
                    log(f"Rust Buffer Size: {len(rgba_bytes)} bytes")
                    max_val = np.max(rgba_np) if len(rgba_np) > 0 else 'Empty'
                    log(f"Buffer Max Value: {max_val}")

                rgba_tensor = torch.from_numpy(rgba_np.reshape(height, width, 4))
                rgba_tensor = rgba_tensor.to(output_images.device)
                
                # Alpha Blending
                alpha = rgba_tensor[:, :, 3:4].float() / 255.0
                overlay_rgb = rgba_tensor[:, :, 0:3].float() / 255.0
                background_rgb = output_images[i]
                
                blended = overlay_rgb * alpha + background_rgb * (1.0 - alpha)
                output_images[i] = blended
                success_count += 1
            except Exception as e:
                if error_count < 5: # Limit error logs
                    log(f"Rust Render Error (Frame {i}): {e}")
                error_count += 1
        
        log(f"Render Complete. Success: {success_count}, Errors: {error_count}")
        return (output_images,)

# Node Registration
NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live (Rust)"
}
