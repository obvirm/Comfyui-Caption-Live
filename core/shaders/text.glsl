// Text rendering shader
// Supports SDF text, stroke, shadow, and color animations

#version 450

layout(local_size_x = 16, local_size_y = 16) in;

// Uniforms
layout(set = 0, binding = 0) uniform TextParams {
    vec4 color;           // Text fill color (RGBA)
    vec4 stroke_color;    // Stroke color (RGBA)
    vec4 shadow_color;    // Shadow color (RGBA)
    vec2 shadow_offset;   // Shadow offset (x, y)
    float shadow_blur;    // Shadow blur radius
    float stroke_width;   // Stroke width in pixels
    float sdf_threshold;  // SDF threshold (typically 0.5)
    float sdf_smoothing;  // SDF edge smoothing
    float time;           // Animation time
    float _padding;
};

// SDF texture
layout(set = 0, binding = 1) uniform sampler2D sdf_texture;

// Output framebuffer
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D output_image;

// Helper: sample SDF with smoothing
float sample_sdf(vec2 uv, float threshold, float smoothing) {
    float distance = texture(sdf_texture, uv).r;
    return smoothstep(threshold - smoothing, threshold + smoothing, distance);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(pixel) / vec2(imageSize(output_image));
    
    // Sample shadow (offset)
    vec2 shadow_uv = uv - shadow_offset / vec2(imageSize(output_image));
    float shadow_alpha = sample_sdf(shadow_uv, sdf_threshold, shadow_blur * 0.01);
    vec4 shadow = shadow_color * shadow_alpha;
    
    // Sample stroke (wider threshold)
    float stroke_alpha = sample_sdf(uv, sdf_threshold - stroke_width * 0.01, sdf_smoothing);
    vec4 stroke = stroke_color * stroke_alpha;
    
    // Sample fill
    float fill_alpha = sample_sdf(uv, sdf_threshold, sdf_smoothing);
    vec4 fill = color * fill_alpha;
    
    // Composite: shadow -> stroke -> fill
    vec4 result = shadow;
    result = mix(result, stroke, stroke.a);
    result = mix(result, fill, fill.a);
    
    imageStore(output_image, pixel, result);
}
