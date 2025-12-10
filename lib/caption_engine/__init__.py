"""
Caption Engine Python bindings
"""

from pathlib import Path
import sys

# Try to import the compiled extension
try:
    from .caption_engine_py import (
        Engine,
        EngineConfig,
        FrameData,
        BackendType,
        Quality,
        Template,
        Position,
        TextStyle,
        Segment,
        EffectType,
        init_effects,
        effect_type_to_string,
        string_to_effect_type,
    )
except ImportError:
    # Fallback: try to find the .pyd file
    _lib_dir = Path(__file__).parent
    _pyd_files = list(_lib_dir.glob("caption_engine_py*.pyd"))
    
    if _pyd_files:
        import importlib.util
        spec = importlib.util.spec_from_file_location("caption_engine_py", _pyd_files[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules["caption_engine_py"] = module
        spec.loader.exec_module(module)
        
        Engine = module.Engine
        EngineConfig = module.EngineConfig
        FrameData = module.FrameData
        BackendType = module.BackendType
        Quality = module.Quality
    else:
        raise ImportError("Caption Engine extension not found. Build with: cmake -B build -DBUILD_PYTHON=ON")


__all__ = [
    'Engine',
    'EngineConfig',
    'FrameData',
    'BackendType',
    'Quality',
    'Template',
    'Position',
    'TextStyle',
    'Segment',
    'EffectType',
    'init_effects',
    'effect_type_to_string',
    'string_to_effect_type',
    'create_engine',
    'render_template',
]


def create_engine(
    width: int = 1920,
    height: int = 1080,
    backend: BackendType = BackendType.Auto,
    quality: Quality = Quality.Final
) -> Engine:
    """Create and configure an engine instance."""
    config = EngineConfig()
    config.width = width
    config.height = height
    config.preferred_backend = backend
    config.quality = quality
    return Engine(config)


def render_template(
    engine: Engine,
    template_json: str,
    time: float
) -> 'numpy.ndarray':
    """Render a template and return as numpy array (H, W, 4)."""
    frame = engine.render_frame(template_json, time)
    return frame.to_numpy()
