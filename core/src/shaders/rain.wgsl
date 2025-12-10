struct FrameUniforms {
    width: u32,
    height: u32,
    time: f32,
    seed: u32,
};

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var<storage, read_write> outputBuffer: array<u32>;

// Simple hashing for GPU randomness
fn hash(p: u32) -> u32 {
    var p_ = p;
    p_ = (p_ + 0x7ed55d16u) + (p_ << 12u);
    p_ = (p_ ^ 0xc761c23cu) ^ (p_ >> 19u);
    p_ = (p_ + 0x165667b1u) + (p_ << 5u);
    p_ = (p_ + 0xd3a2646cu) ^ (p_ << 9u);
    p_ = (p_ + 0xfd7046c5u) + (p_ << 3u);
    p_ = (p_ ^ 0xb55a4f09u) ^ (p_ >> 16u);
    return p_;
}

fn rand(id: u32) -> f32 {
    return f32(hash(id)) / 4294967295.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= uniforms.width || y >= uniforms.height) { return; }

    // Seed depends on pixel + global seed + time (for animation)
    // For deterministic rain snapshot, use 0 time or fixed time.
    let pixel_index = y * uniforms.width + x;
    
    // Falling rain logic
    // We map pixel to a uv coordinate
    let u = f32(x) / f32(uniforms.width);
    let v = f32(y) / f32(uniforms.height);
    
    // Animate Y with time
    let drop_speed = 2.0;
    let time_offset = uniforms.time * drop_speed;
    
    // Grid cells for droplets
    let cols = 50.0;
    let rows = 20.0;
    
    let cell_x = floor(u * cols);
    let cell_y = floor((v + time_offset) * rows);
    
    // Hash cell to decide if it has a drop
    let cell_hash = hash(u32(cell_x) + u32(cell_y) * 100u + uniforms.seed);
    
    var color: u32 = 0xFF000000u; // Black, 100% alpha (ABGR)
    // 0xAABBGGRR in hex literal usually, but here u32(r) | ...
    
    // If has drop (prob 5%)
    if ((cell_hash % 100u) < 5u) {
        // White drop
        color = 0xFFFFFFFFu;
    } else {
        // Gradient Background (Blue-ish)
        let b = u32(v * 100.0);
        color = 0xFF000000u | (b << 16); 
    }
    
    outputBuffer[pixel_index] = color;
}
