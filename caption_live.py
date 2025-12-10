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
    """Load the C++ caption engine"""
    global caption_backend, cpp_engine
    if caption_backend is not None:
        return caption_backend
    
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # C++ caption_engine_py only
    try:
        import caption_engine_py
        cpp_engine = caption_engine_py.Engine()
        caption_backend = caption_engine_py
        print(f"✅ C++ Caption Engine Loaded! Backend: {cpp_engine.current_backend()}")
        return caption_backend
    except ImportError as e:
        print(f"❌ C++ Caption Engine not found: {e}")
        print("   Please build the C++ engine: cd caption_engine_cpp && ./build_python.ps1")
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
                       style, highlight_color, text_color, stroke_color, 
                       stroke_width, pos_x, pos_y, segments_str):
        """Build template JSON for caption_engine"""
        # Parse segments
        segments = []
        try:
            parsed = json.loads(segments_str.replace("'", '"'))
            if isinstance(parsed, list) and len(parsed) > 0:
                segments = parsed
        except:
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
                    "color": text_color,
                    "stroke_color": stroke_color,
                    "stroke_width": stroke_width
                },
                "position": {"x": pos_x, "y": pos_y},
                "animation": animation
            }]
        }
        
        return json.dumps(template)

    def process(self, images, text, font_size, duration, aspect_ratio="16:9",
                style="box", highlight_color="#39E55F", text_color="#FFFFFF",
                stroke_color="#000000", stroke_width=4.0, pos_x=0.5, pos_y=0.8, 
                segments="[]"):
        
        backend = load_caption_backend()
        if backend is None:
            print("❌ No caption backend available!")
            return (images,)

        batch_size, height, width, _ = images.shape
        output_images = []

        # Build template
        template_json = self.build_template(
            width, height, duration, text, font_size,
            style, highlight_color, text_color, stroke_color,
            stroke_width, pos_x, pos_y, segments
        )

        # Render loop
        for i in range(batch_size):
            progress = i / max(batch_size - 1, 1)
            current_time = progress * duration

            try:
                # Start with input image
                input_img = images[i].clone()
                if input_img.shape[2] == 3:  # RGB -> RGBA
                    input_img = torch.cat([input_img, torch.ones((height, width, 1))], dim=2)
                
                result = input_img

                # Render caption using C++ caption_engine
                if cpp_engine is not None:
                    frame = cpp_engine.render_frame(template_json, current_time)
                    caption_np = np.frombuffer(bytes(frame.pixels), dtype=np.uint8)
                    caption_np = caption_np.reshape((frame.height, frame.width, 4)).astype(np.float32) / 255.0
                    caption_tensor = torch.from_numpy(caption_np)
                else:
                    raise RuntimeError("C++ Caption Engine not loaded!")
                
                # Alpha composite caption over input
                alpha = caption_tensor[:, :, 3:4]
                result = caption_tensor * alpha + result * (1 - alpha)
                
                output_images.append(result)
                
            except Exception as e:
                print(f"Render Error Frame {i}: {e}")
                import traceback
                traceback.print_exc()
                output_images.append(images[i])

        return (torch.stack(output_images),)


class CaptionLiveGPUNode:
    @classmethod
    def INPUT_TYPES(cls):
        # List available effect definitions
        effects_dir = os.path.join(os.path.dirname(__file__), "caption_engine", "src", "effects", "definitions")
        effect_files = ["custom"]
        if os.path.exists(effects_dir):
            effect_files += [f.replace(".yaml", "") for f in os.listdir(effects_dir) if f.endswith(".yaml")]

        return {
            "required": {
                "images": ("IMAGE",),
                "effect_name": (effect_files, {"default": "chromatic_aberration"}),
            },
            "optional": {
                "custom_yaml": ("STRING", {"multiline": True, "default": "", "placeholder": "Paste custom YAML here if effect_name is 'custom'"}),
                "param1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "param2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "param3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "Caption Live/GPU"

    def apply_effect(self, images, effect_name, custom_yaml="", param1=0.5, param2=0.0, param3=0.0):
        backend = load_caption_backend()
        if not hasattr(backend, 'GPUManager'):
            print("❌ GPU Backend not available")
            return (images,)

        try:
            # Determine YAML source
            yaml_source = custom_yaml
            if effect_name != "custom":
                yaml_path = os.path.join(os.path.dirname(__file__), "caption_engine", "src", "effects", "definitions", f"{effect_name}.yaml")
                if os.path.exists(yaml_path):
                    with open(yaml_path, "r") as f:
                        yaml_source = f.read()
                else:
                    print(f"⚠️ Effect file not found: {yaml_path}")
            
            if not yaml_source.strip():
                print("⚠️ No effect definition provided")
                return (images,)

            # Initialize GPU Context
            manager = backend.GPUManager()
            
            # Compile & Load
            # print("⚙️ Compiling Effect...")
            wgsl = backend.compile_effect(yaml_source)
            manager.load_effect(wgsl)
            
            results = []
            batch_size, height, width, channels = images.shape
            
            # print(f"🚀 Processing {batch_size} frames on GPU...")
            
            for i in range(batch_size):
                # Ensure RGBA (4 channels)
                img = images[i]
                if channels == 3:
                    # Pad with Alpha=1.0
                    alpha = torch.ones((height, width, 1), dtype=img.dtype, device=img.device)
                    img = torch.cat((img, alpha), dim=2)
                
                # Parameters
                params = [param1, param2, param3] 
                
                # --- Zero-Copy Optimization ---
                # Convert tensor to bytes (must be contiguous)
                # Note: We ensure float32 type
                input_bytes = img.contiguous().cpu().numpy().tobytes()
                
                # Execute on GPU with Buffer (Zero-Copy In)
                # Returns raw bytes (Vec<u8>)
                output_bytes = manager.execute_buffer(input_bytes, params, width, height)
                
                # Zero-Copy Out: Create tensor from bytes
                # frombuffer creates a read-only tensor sharing memory, clone to make it writable/standard
                # We need to specify count explicitly or reshape
                out_tensor = torch.frombuffer(bytearray(output_bytes), dtype=torch.float32).reshape(height, width, 4).clone()
                
                # If input was RGB, discard alpha
                if channels == 3:
                    out_tensor = out_tensor[:, :, :3]
                    
                results.append(out_tensor)
                
            return (torch.stack(results),)
            
        except Exception as e:
            print(f"❌ GPU Execution Failed: {e}")
            import traceback
            traceback.print_exc()
            return (images,)


NODE_CLASS_MAPPINGS = {
    "CaptionLiveNode": CaptionLiveNode,
    "CaptionLiveGPUNode": CaptionLiveGPUNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaptionLiveNode": "Caption Live",
    "CaptionLiveGPUNode": "Caption Live (GPU Compute)"
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