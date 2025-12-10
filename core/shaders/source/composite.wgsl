// Compositing Shader (WGSL)
// Final frame compositing with alpha blending

struct CompositeUniforms {
    outputSize: vec2<f32>,
    time: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: CompositeUniforms;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var overlayTexture: texture_2d<f32>;
@group(0) @binding(3) var texSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
};

// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // Generate fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    
    var texCoords = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );
    
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
    output.texCoord = texCoords[vertexIndex];
    return output;
}

// Porter-Duff "over" compositing operator
fn composite_over(backdrop: vec4<f32>, source: vec4<f32>) -> vec4<f32> {
    let srcA = source.a;
    let dstA = backdrop.a;
    
    let outA = srcA + dstA * (1.0 - srcA);
    
    if (outA < 0.001) {
        return vec4<f32>(0.0);
    }
    
    let outRGB = (source.rgb * srcA + backdrop.rgb * dstA * (1.0 - srcA)) / outA;
    return vec4<f32>(outRGB, outA);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let backdrop = textureSample(inputTexture, texSampler, input.texCoord);
    let overlay = textureSample(overlayTexture, texSampler, input.texCoord);
    
    return composite_over(backdrop, overlay);
}

// === VARIANT: Compute shader for compositing (more flexible) ===

@group(0) @binding(0) var inputImage: texture_2d<f32>;
@group(0) @binding(1) var overlayImage: texture_2d<f32>;
@group(0) @binding(2) var outputImage: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn cs_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(inputImage);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let coord = vec2<i32>(gid.xy);
    let backdrop = textureLoad(inputImage, coord, 0);
    let overlay = textureLoad(overlayImage, coord, 0);
    
    let result = composite_over(backdrop, overlay);
    textureStore(outputImage, coord, result);
}
