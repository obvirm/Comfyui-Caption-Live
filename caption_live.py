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
                "gpu_acceleration": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "caption-live"

    def render(self, images, segments, fps, font_size, highlight_color, text_color, font_path, aspect_ratio, pos_x, pos_y, style, gpu_acceleration, mask_optional=None):
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
        log(f"Input Segments (raw): {original_font_path}")

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
        
        width = int(images.shape[2])
        height = int(images.shape[1])
        
        baseline_width = 210.0
        scale_factor = width / baseline_width
        
        scaled_font_size = float(font_size) * scale_factor
        scaled_pos_x = float(pos_x) * scale_factor
        scaled_pos_y = float(pos_y) * scale_factor
        
        log(f"Rendering Params: W={width}, H={height}, Scale={scale_factor:.2f}, FontSize={scaled_font_size:.1f}")
        
        try:
            # --- CHUNKING & MEMORY OPTIMIZATION ---
            import gc
            
            chunk_size = 50
            total_frames = images.shape[0]
            output_tensors = []
            
            log(f"Starting Batch Render. Total Frames: {total_frames}, Chunk Size: {chunk_size}")

            for start_idx in range(0, total_frames, chunk_size):
                end_idx = min(start_idx + chunk_size, total_frames)
                log(f"Processing Chunk {start_idx}-{end_idx}...")
                
                # 1. Prepare Chunk
                chunk_images = images[start_idx:end_idx]
                
                # Ensure RGBA
                is_rgb_input = chunk_images.shape[3] == 3
                if is_rgb_input:
                    alpha = torch.ones((chunk_images.shape[0], chunk_images.shape[1], chunk_images.shape[2], 1), device=chunk_images.device)
                    chunk_images = torch.cat((chunk_images, alpha), dim=3)
                    del alpha # Cleanup immediately
                
                # Convert to bytes
                images_uint8 = (chunk_images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                images_bytes = [frame.tobytes() for frame in images_uint8]
                
                # Cleanup intermediate tensors
                del chunk_images
                del images_uint8
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 2. Call Rust
                if gpu_acceleration:
                    log("Using GPU Acceleration")
                    output_bytes_list = rust_caption.render_gpu(
                        images_bytes, 
                        width, 
                        height, 
                        fps, 
                        segments_json, 
                        font_path, 
                        style, 
                        highlight_color, 
                        text_color, 
                        scaled_pos_x, 
                        scaled_pos_y, 
                        scaled_font_size
                    )
                else:
                    output_bytes_list = rust_caption.render_tiktok_batch(
                        images_bytes, 
                        width, 
                        height, 
                        fps, 
                        segments_json, 
                        font_path, 
                        style, 
                        highlight_color, 
                        text_color, 
                        scaled_pos_x, 
                        scaled_pos_y, 
                        scaled_font_size
                    )
                
                # Cleanup input bytes
                del images_bytes
                
                # 3. Reconstruct Tensor
                chunk_tensors = []
                for raw_bytes in output_bytes_list:
                    if isinstance(raw_bytes, list):
                        img_np = np.array(raw_bytes, dtype=np.uint8).reshape(height, width, 4)
                    else:
                        img_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, 4)
                    
                    img_tensor = torch.from_numpy(img_np).float() / 255.0
                    chunk_tensors.append(img_tensor)
                
                # Cleanup Rust output
                del output_bytes_list
                
                # Stack Chunk
                chunk_final = torch.stack(chunk_tensors).to(images.device)
                
                # Restore RGB if needed
                if is_rgb_input:
                    chunk_final = chunk_final[:, :, :, :3]
                
                output_tensors.append(chunk_final)
                
                # Aggressive Cleanup after each chunk
                del chunk_tensors
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # Concatenate all chunks
            final_output = torch.cat(output_tensors, dim=0)
            
            log(f"Render Complete. Final Shape: {final_output.shape}")
            return (final_output,)

        except Exception as e:
            log(f"Rust Batch Render Error: {e}")
            import traceback
            log(traceback.format_exc())
            return (images,)

# Node Registration
NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live (Rust)"
}
