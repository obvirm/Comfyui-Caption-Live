"""Test script for caption_engine_py"""
import caption_engine_py as ce

print('1. Import OK')

# Test EngineConfig
cfg = ce.EngineConfig()
print(f'2. EngineConfig OK')

# Test BackendType
print(f'3. BackendType.Auto = {ce.BackendType.Auto}')

# Test Quality (use actual enum values)
print(f'4. Quality values = {list(ce.Quality.__members__.keys())}')

# Test Engine creation
cfg.quality = ce.Quality.High
engine = ce.Engine(cfg)
print(f'5. Engine created: {engine}')

# Test backend
backend = engine.current_backend()
print(f'6. Current backend: {backend}')

# Test render
template = '{"canvas":{"width":1920,"height":1080},"duration":5.0,"fps":60.0,"layers":[]}'
frame = engine.render_frame(template, 0.0)
print(f'7. Frame rendered: {frame.width}x{frame.height}')
print(f'8. Pixels size: {len(frame.pixels)}')

print()
print('=== ALL TESTS PASSED ===')
