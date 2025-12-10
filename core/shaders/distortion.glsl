// Distortion effects shader
// Supports glitch, chromatic aberration, wave distortion

#version 450

layout(local_size_x = 16, local_size_y = 16) in;

// Effect type enum
#define EFFECT_NONE 0
#define EFFECT_GLITCH 1
#define EFFECT_CHROMATIC 2
#define EFFECT_WAVE 3
#define EFFECT_VHS 4
#define EFFECT_PIXELATE 5

layout(set = 0, binding = 0) uniform DistortParams {
    int effect_type;
    float time;
    float intensity;
    float param1;        // Effect-specific
    float param2;        // Effect-specific
    float param3;        // Effect-specific
    vec2 center;         // Effect center (normalized)
    ivec2 resolution;    // Output resolution
    uint seed;           // RNG seed
    uint _pad;
};

layout(set = 0, binding = 1) uniform sampler2D input_texture;
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D output_image;

// Pseudo-random function
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Glitch effect
vec4 apply_glitch(vec2 uv) {
    float glitch_prob = param1;
    float rgb_split = param2;
    
    // Random glitch bands
    float band = floor(uv.y * 20.0 + time * 10.0);
    float rand = hash(vec2(band, floor(time * 60.0)));
    
    if (rand < glitch_prob * intensity) {
        // Horizontal offset
        float offset = (hash(vec2(band, time)) * 2.0 - 1.0) * intensity * 0.1;
        uv.x += offset;
        
        // RGB split
        vec2 offset_r = vec2(rgb_split * intensity * 0.01, 0.0);
        vec2 offset_b = -offset_r;
        
        float r = texture(input_texture, uv + offset_r).r;
        float g = texture(input_texture, uv).g;
        float b = texture(input_texture, uv + offset_b).b;
        
        return vec4(r, g, b, 1.0);
    }
    
    return texture(input_texture, uv);
}

// Chromatic aberration
vec4 apply_chromatic(vec2 uv) {
    vec2 dir = uv - center;
    float dist = length(dir);
    
    if (dist > 0.001) {
        dir = normalize(dir);
        float offset = dist * intensity;
        
        vec2 uv_r = uv - dir * offset;
        vec2 uv_b = uv + dir * offset;
        
        float r = texture(input_texture, uv_r).r;
        float g = texture(input_texture, uv).g;
        float b = texture(input_texture, uv_b).b;
        
        return vec4(r, g, b, 1.0);
    }
    
    return texture(input_texture, uv);
}

// Wave distortion
vec4 apply_wave(vec2 uv) {
    float amplitude_x = param1 / float(resolution.x);
    float amplitude_y = param1 / float(resolution.y);
    float frequency = param2;
    float speed = param3;
    
    float phase = time * speed;
    
    float dx = sin(uv.y * frequency * 6.28318 + phase) * amplitude_x * intensity;
    float dy = sin(uv.x * frequency * 6.28318 + phase) * amplitude_y * intensity;
    
    return texture(input_texture, uv + vec2(dx, dy));
}

// VHS effect
vec4 apply_vhs(vec2 uv) {
    float scanline_strength = param1;
    float noise_strength = param2;
    
    // Scanlines
    float scanline = 1.0 - scanline_strength * mod(floor(uv.y * float(resolution.y)), 2.0);
    
    // Noise
    float noise = (hash(uv + time) * 2.0 - 1.0) * noise_strength * intensity;
    
    // Color bleed (horizontal blur on red channel)
    float r = texture(input_texture, uv + vec2(0.002 * intensity, 0.0)).r;
    float g = texture(input_texture, uv).g;
    float b = texture(input_texture, uv - vec2(0.002 * intensity, 0.0)).b;
    
    vec3 color = vec3(r, g, b) * scanline + noise;
    
    return vec4(color, 1.0);
}

// Pixelate effect
vec4 apply_pixelate(vec2 uv) {
    int block_size = int(param1 * intensity);
    if (block_size < 1) block_size = 1;
    
    ivec2 pixel = ivec2(uv * vec2(resolution));
    pixel = (pixel / block_size) * block_size + block_size / 2;
    
    vec2 snapped_uv = vec2(pixel) / vec2(resolution);
    
    return texture(input_texture, snapped_uv);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= resolution.x || pixel.y >= resolution.y) return;
    
    vec2 uv = (vec2(pixel) + 0.5) / vec2(resolution);
    
    vec4 result;
    
    switch (effect_type) {
        case EFFECT_GLITCH:
            result = apply_glitch(uv);
            break;
        case EFFECT_CHROMATIC:
            result = apply_chromatic(uv);
            break;
        case EFFECT_WAVE:
            result = apply_wave(uv);
            break;
        case EFFECT_VHS:
            result = apply_vhs(uv);
            break;
        case EFFECT_PIXELATE:
            result = apply_pixelate(uv);
            break;
        default:
            result = texture(input_texture, uv);
    }
    
    imageStore(output_image, pixel, result);
}
