// Basic Compute Shader for Caption Engine
// Fills buffer with a solid color or pattern for testing

struct FrameUniforms {
    width: u32,
    height: u32,
    time: f32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>; // RGBA8 packed as u32

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }
    
    let index = y * uniforms.width + x;
    
    // Generate color based on UV
    let u = f32(x) / f32(uniforms.width);
    let v = f32(y) / f32(uniforms.height);
    
    let r = u32(u * 255.0);
    let g = u32(v * 255.0);
    let b = u32((sin(uniforms.time) * 0.5 + 0.5) * 255.0);
    let a = 255u;
    
    // Pack RGBA8 (Little Endian: ABGR in u32)
    // format: R is LSB
    let color = r | (g << 8) | (b << 16) | (a << 24);
    
    outputBuffer[index] = color;
}
