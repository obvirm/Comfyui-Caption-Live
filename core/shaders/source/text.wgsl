// SDF Text Rendering Shader (WGSL)
// Renders text using Signed Distance Fields for crisp edges at any scale

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct TextUniforms {
    transform: mat4x4<f32>,
    color: vec4<f32>,
    outlineColor: vec4<f32>,
    outlineWidth: f32,
    softness: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: TextUniforms;
@group(0) @binding(1) var sdfTexture: texture_2d<f32>;
@group(0) @binding(2) var sdfSampler: sampler;

// Vertex shader - quad for each glyph
@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) color: vec4<f32>
) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.transform * vec4<f32>(position, 0.0, 1.0);
    output.texCoord = texCoord;
    output.color = color * uniforms.color;
    return output;
}

// Fragment shader - SDF rendering with anti-aliasing
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample SDF texture
    let distance = textureSample(sdfTexture, sdfSampler, input.texCoord).r;
    
    // SDF threshold (0.5 = edge)
    let threshold = 0.5;
    
    // Calculate anti-aliased alpha using screen-space derivatives
    let screenSpaceWidth = fwidth(distance);
    let alpha = smoothstep(threshold - screenSpaceWidth, threshold + screenSpaceWidth, distance);
    
    // Outline calculation
    let outlineThreshold = threshold - uniforms.outlineWidth;
    let outlineAlpha = smoothstep(
        outlineThreshold - screenSpaceWidth, 
        outlineThreshold + screenSpaceWidth, 
        distance
    );
    
    // Blend outline and fill colors
    var finalColor: vec4<f32>;
    if (uniforms.outlineWidth > 0.0) {
        // Has outline
        let fillColor = input.color * alpha;
        let strokeColor = uniforms.outlineColor * (outlineAlpha - alpha);
        finalColor = fillColor + strokeColor;
    } else {
        // No outline
        finalColor = vec4<f32>(input.color.rgb, input.color.a * alpha);
    }
    
    return finalColor;
}

// === VARIATION: Box highlight behind text ===

struct BoxUniforms {
    transform: mat4x4<f32>,
    boxColor: vec4<f32>,
    cornerRadius: f32,
    padding: f32,
    time: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> boxUniforms: BoxUniforms;

@vertex
fn vs_box(
    @location(0) position: vec2<f32>,
    @location(1) texCoord: vec2<f32>
) -> VertexOutput {
    var output: VertexOutput;
    output.position = boxUniforms.transform * vec4<f32>(position, 0.0, 1.0);
    output.texCoord = texCoord;
    output.color = boxUniforms.boxColor;
    return output;
}

// Rounded rectangle SDF
fn sdRoundedBox(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_box(input: VertexOutput) -> @location(0) vec4<f32> {
    // Map texCoord to [-1, 1] range
    let p = input.texCoord * 2.0 - 1.0;
    
    // Calculate SDF for rounded rectangle
    let boxSize = vec2<f32>(1.0 - boxUniforms.padding, 1.0 - boxUniforms.padding);
    let d = sdRoundedBox(p, boxSize, boxUniforms.cornerRadius);
    
    // Anti-aliased edge
    let aa = fwidth(d);
    let alpha = 1.0 - smoothstep(-aa, aa, d);
    
    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
