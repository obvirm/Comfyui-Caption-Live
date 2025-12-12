# CaptionEngine Architecture

## ⚠️ CRITICAL: 100% GPU-Only Pipeline

**Canvas 2D is NEVER used in this engine.**

---

## Rendering Backends

| Platform | API | Status |
|----------|-----|--------|
| Desktop/Server | **Vulkan** | ✅ Primary |
| Windows | **DirectX 12** | ✅ Native |
| Apple | **Metal** | ✅ Native |
| Browser | **WebGPU** | ✅ Required |
| NVIDIA | **CUDA** | ✅ Compute |

## ❌ NOT USED (Ever)

- ❌ Canvas 2D
- ❌ CanvasRenderingContext2D
- ❌ ctx.fillText()
- ❌ ctx.drawImage()
- ❌ JavaScript CPU rendering

---

## Why No Canvas 2D?

### Performance
| Task | Canvas 2D | GPU | Speedup |
|------|-----------|-----|---------|
| 1000 chars | 50-100ms | 1-2ms | **50x** |
| 10k particles | 500ms+ | 5-10ms | **100x** |
| Liquid sim | Impossible | 10-20ms | **∞** |

### Quality
- **Canvas 2D**: Pixelated, inconsistent, limited effects
- **GPU**: SDF text, advanced shaders, crisp at any scale

### Consistency
- **Canvas 2D**: Different per browser/OS
- **GPU**: Identical output everywhere

---

## GPU Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CaptionEngine Core                        │
│                     (C++ / WASM)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              GPU Backend Abstraction                 │   │
│  │                  (gpu/backend.hpp)                   │   │
│  └───────────┬───────────┬───────────┬────────────────┘   │
│              │           │           │                     │
│      ┌───────▼───┐ ┌─────▼─────┐ ┌──▼────┐ ┌─────────┐   │
│      │  Vulkan   │ │  WebGPU   │ │ DX12  │ │  Metal  │   │
│      │ (Server)  │ │ (Browser) │ │ (Win) │ │ (Apple) │   │
│      └───────────┘ └───────────┘ └───────┘ └─────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     GPU Hardware                            │
│                                                             │
│    NVIDIA RTX    AMD RDNA    Intel Arc    Apple Silicon    │
└─────────────────────────────────────────────────────────────┘
```

---

## Text Rendering Pipeline (SDF)

```
Font File (TTF/OTF)
       │
       ▼
┌──────────────────┐
│    FreeType      │  ← Glyph extraction
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  GPU Compute     │  ← SDF generation (compute shader)
│  SDF Generator   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  SDF Texture     │  ← GPU texture atlas
│  Atlas           │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  GPU Fragment    │  ← SDF rendering (fragment shader)
│  Shader Render   │
└────────┬─────────┘
         │
         ▼
   Crisp Text Output
   (Any resolution)
```

---

## Effect Pipeline

```wgsl
// GPU Compute Shader (WGSL)
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    // GPU-parallel processing
    let color = compute_effect(x, y, uniforms.time);
    
    output[gid.y * width + gid.x] = pack_rgba(color);
}
```

---

## Shader Languages

| Platform | Language | Format |
|----------|----------|--------|
| Vulkan | GLSL | SPIR-V |
| WebGPU | WGSL | WGSL |
| DX12 | HLSL | DXIL |
| Metal | MSL | Metallib |
| CUDA | CUDA C++ | PTX |

Cross-compilation handled by shader compiler module.

---

## Browser Requirements

**Minimum**: Chrome 113+ / Edge 113+ / Firefox Nightly

**Required WebGPU Features**:
- Compute shaders
- Storage buffers
- Texture sampling

**If WebGPU not available**: Show error message. **NO Canvas 2D fallback.**

---

## Build Configuration

```cmake
# GPU Backends
option(ENABLE_VULKAN "Vulkan backend" ON)
option(ENABLE_DX12 "DirectX 12 backend" ON)   # Windows
option(ENABLE_METAL "Metal backend" ON)        # macOS
option(ENABLE_CUDA "CUDA compute" ON)          # NVIDIA

# NO Canvas 2D option exists - it's not supported!
```

---

## Consistency Guarantee

```
Native Build (Vulkan/DX12) ──┐
                             ├──► Same Output
WASM Build (WebGPU) ─────────┘

Deterministic:
- Fixed-point math where needed
- Reproducible RNG
- Frame checksums
- Golden reference validation
```
