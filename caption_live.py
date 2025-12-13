"""
Caption Live - ComfyUI Node
Uses C++ caption_engine for deterministic, cross-platform rendering.
"""
import torch
import numpy as np
import json
import os
import sys
from PIL import Image
import io
import server
from aiohttp import web

# Dynamic Import Logic
caption_backend = None
cpp_engine = None  # New C++ engine instance

def load_caption_backend():
    """Load the unified GPU render pipeline (Vulkan > WebGPU > Legacy)"""
    global caption_backend, cpp_engine
    if caption_backend is not None:
        return caption_backend
    
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Try unified GPU pipeline first (Vulkan + CUDA)
    try:
        import caption_engine_unified as unified
        pipeline = unified.get_pipeline()
        
        # Initialize with default target
        target = unified.RenderTarget()
        target.width = 1920
        target.height = 1080
        target.fps = 60.0
        
        if pipeline.initialize(target):
            caption_backend = unified
            cpp_engine = pipeline
            print(f"✅ Unified GPU Pipeline Loaded!")
            print(f"   Backend: {pipeline.backend_name()}")
            print(f"   CUDA Acceleration: {'Yes' if pipeline.has_cuda() else 'No'}")
            return caption_backend
    except ImportError:
        print("ℹ️ Unified pipeline not available, trying legacy...")
    except Exception as e:
        print(f"⚠️ Unified pipeline init failed: {e}")
    
    # Fallback to legacy C++ caption_engine_py
    try:
        import caption_engine_py
        cpp_engine = caption_engine_py.Engine()
        caption_backend = caption_engine_py
        print(f"✅ Legacy C++ Engine Loaded! Backend: {cpp_engine.current_backend()}")
        return caption_backend
    except ImportError as e:
        print(f"❌ No caption engine found: {e}")
        print("   Build unified: cmake --build core/build --config Release")
        print("   Or legacy: cd core && ./build_python.ps1")
        return None


class CaptionLiveNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "Hello World", "multiline": True}),
                "font_size": ("FLOAT", {"default": 40.0, "min": 10.0, "max": 500.0}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
            },
            "optional": {
                "style": (["box", "typewriter", "bounce", "colored"], {"default": "box"}),
                "font_path": ("STRING", {"default": "", "multiline": False}),
                "highlight_color": ("STRING", {"default": "#39E55F"}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "stroke_color": ("STRING", {"default": "#000000"}),
                "stroke_width": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0}),
                "pos_x": ("FLOAT", {"default": 0.5, "min": -50.0, "max": 50.0, "step": 0.001}),
                "pos_y": ("FLOAT", {"default": 0.8, "min": -50.0, "max": 50.0, "step": 0.001}),
                "segments": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Caption Live"

    def build_template(self, width, height, duration, text, font_size, 
                       style, font_path, highlight_color, text_color, stroke_color, 
                       stroke_width, pos_x, pos_y, segments_str):
        """Build template JSON for caption_engine"""
        # Parse segments
        segments = []
        # Parse segments with WhisperX support
        segments = []
        try:
            # Handle potential single quotes from some inputs
            cleaned_str = segments_str.replace("'", '"')
            parsed = json.loads(cleaned_str)

            final_segments = []
            
            # Handle full WhisperX object (root has "segments")
            if isinstance(parsed, dict) and "segments" in parsed:
                parsed = parsed["segments"]

            # Process list of segments/words
            if isinstance(parsed, list):
                for item in parsed:
                    # Priority: Extract "words" if available (Word-Level Timing)
                    if "words" in item and isinstance(item["words"], list):
                        for word in item["words"]:
                            final_segments.append({
                                "text": word.get("word", "").strip(),
                                "start": float(word.get("start", 0)),
                                "end": float(word.get("end", 0))
                            })
                    # Fallback: Use segment text if no words array
                    elif "text" in item and "start" in item:
                        final_segments.append({
                            "text": item.get("text", "").strip(),
                            "start": float(item.get("start", 0)),
                            "end": float(item.get("end", 0))
                        })
            
            if final_segments:
                segments = final_segments

        except Exception as e:
            print(f"⚠️ JSON Parse Warning: {e}")
            pass
        
        # If segments empty, build from text words (same as frontend)
        if not segments:
            words = text.split()
            if words:
                time_per_word = duration / len(words)
                for i, word in enumerate(words):
                    segments.append({
                        "text": word,
                        "start": i * time_per_word,
                        "end": (i + 1) * time_per_word
                    })
        
        # Build animation based on style
        animation = None
        if style == "box":
            animation = {
                "type": "box_highlight",
                "segments": segments,
                "box_color": highlight_color,
                "box_radius": 8.0,
                "box_padding": 8.0
            }
        elif style == "typewriter":
            animation = {"type": "typewriter", "segments": segments}
        elif style == "bounce":
            animation = {"type": "bounce", "segments": segments, "intensity": 1.2}
        elif style == "colored":
            animation = {"type": "colored", "segments": segments, "active_color": highlight_color}
        
        # Font scaling - same as frontend (relative to 1080p height)
        scale_factor = height / 1080.0
        scaled_font_size = font_size * scale_factor
        # Scale stroke_width proportionally with font for visual consistency
        scaled_stroke_width = stroke_width * scale_factor
        
        # Build content from segments (same as frontend)
        content = " ".join([s.get("text", "") for s in segments]) if segments else text
        
        template = {
            "canvas": {"width": width, "height": height},
            "duration": duration,
            "fps": 60.0,
            "layers": [{
                "type": "text",
                "content": content,
                "style": {
                    "font_size": scaled_font_size,
                    "font_path": font_path if font_path else None,
                    "color": text_color,
                    "stroke_color": stroke_color,
                    "stroke_width": scaled_stroke_width  # Pre-scaled, C++ won't scale again
                },
                "position": {"x": pos_x, "y": pos_y},
                "animation": animation
            }]
        }
        
        return json.dumps(template)

    def process(self, images, text, font_size, duration, aspect_ratio="16:9",
                style="box", font_path="", highlight_color="#39E55F", text_color="#FFFFFF",
                stroke_color="#000000", stroke_width=4.0, pos_x=0.5, pos_y=0.8, 
                segments="[]"):
        
        backend = load_caption_backend()
        if backend is None:
            print("❌ No caption backend available!")
            return (images,)

        batch_size, height, width, channels = images.shape
        output_images = []

        # Build template JSON (Scene Description)
        template_json = self.build_template(
            width, height, duration, text, font_size,
            style, font_path, highlight_color, text_color, stroke_color,
            stroke_width, pos_x, pos_y, segments
        )

        # Render loop - using unified process_frame API
        for i in range(batch_size):
            progress = i / max(batch_size - 1, 1)
            current_time = progress * duration

            try:
                # Get input frame as contiguous numpy array
                input_np = images[i].cpu().numpy().astype(np.float32)
                
                # Ensure RGBA (4 channels)
                if input_np.shape[2] == 3:
                    alpha_channel = np.ones((height, width, 1), dtype=np.float32)
                    input_np = np.concatenate([input_np, alpha_channel], axis=2)
                
                # Use unified process_frame API (compositing in C++)
                # This is faster as C++ handles both rendering and compositing
                if hasattr(backend, 'process_frame'):
                    # New unified API: C++ does render + composite
                    result_np = backend.process_frame(template_json, current_time, input_np)
                    result_tensor = torch.from_numpy(result_np)
                else:
                    # Fallback: Old API with Python compositing
                    frame = cpp_engine.render_frame(template_json, current_time)
                    caption_np = np.frombuffer(bytes(frame.pixels), dtype=np.uint8)
                    caption_np = caption_np.reshape((frame.height, frame.width, 4)).astype(np.float32) / 255.0
                    
                    # Alpha composite in Python
                    alpha = caption_np[:, :, 3:4]
                    result_np = caption_np * alpha + input_np * (1 - alpha)
                    result_tensor = torch.from_numpy(result_np)
                
                # Convert back to RGB if input was RGB
                if channels == 3:
                    result_tensor = result_tensor[:, :, :3]
                
                output_images.append(result_tensor)
                
            except Exception as e:
                print(f"Render Error Frame {i}: {e}")
                import traceback
                traceback.print_exc()
                output_images.append(images[i])

        return (torch.stack(output_images),)


# Single unified node with C++ backend
NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live",
}

# ============================================================================
# API Routes
# ============================================================================

@server.PromptServer.instance.routes.get("/caption-live/effects")
async def get_effects(request):
    """List all available effect definitions"""
    effects_dir = os.path.join(os.path.dirname(__file__), "caption_engine", "src", "effects", "definitions")
    effects = {}
    
    if os.path.exists(effects_dir):
        for f in os.listdir(effects_dir):
            if f.endswith(".yaml"):
                name = f.replace(".yaml", "")
                path = os.path.join(effects_dir, f)
                with open(path, "r") as file:
                    effects[name] = file.read()
                    
    return web.json_response(effects)


# --- Standalone Editor Routes ---

@server.PromptServer.instance.routes.get("/caption-live/editor")
async def serve_editor(request):
    """Serve the standalone Caption Live Editor"""
    editor_dir = os.path.join(os.path.dirname(__file__), "web", "editor")
    index_path = os.path.join(editor_dir, "index.html")
    
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return web.Response(text=content, content_type="text/html")
    else:
        return web.Response(text="Editor not found", status=404)


@server.PromptServer.instance.routes.get("/caption-live/editor/{filename}")
async def serve_editor_files(request):
    """Serve editor static files (CSS, JS)"""
    filename = request.match_info.get("filename", "")
    editor_dir = os.path.join(os.path.dirname(__file__), "web", "editor")
    file_path = os.path.join(editor_dir, filename)
    
    # Security: prevent directory traversal
    if ".." in filename or not os.path.exists(file_path):
        return web.Response(text="File not found", status=404)
    
    # Determine content type
    content_type = "text/plain"
    if filename.endswith(".css"):
        content_type = "text/css"
    elif filename.endswith(".js"):
        content_type = "application/javascript"
    elif filename.endswith(".html"):
        content_type = "text/html"
    elif filename.endswith(".json"):
        content_type = "application/json"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return web.Response(text=content, content_type=content_type)