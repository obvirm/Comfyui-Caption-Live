// Particle system compute shader
// Simulates and renders particle effects

#version 450

layout(local_size_x = 256) in;

// Particle structure
struct Particle {
    vec2 position;
    vec2 velocity;
    vec2 acceleration;
    float rotation;
    float angular_velocity;
    float scale;
    float life;
    float max_life;
    uint color;
};

// Buffers
layout(set = 0, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

layout(set = 0, binding = 1) uniform SimParams {
    float delta_time;
    float gravity;
    float drag;
    float scale_start;
    float scale_end;
    uint particle_count;
    uint _pad0;
    uint _pad1;
};

// RNG state
uint rng_state;

uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float() {
    rng_state = pcg_hash(rng_state);
    return float(rng_state) / 4294967295.0;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= particle_count) return;
    
    // Initialize RNG
    rng_state = idx * 1234567u + uint(delta_time * 60.0);
    
    Particle p = particles[idx];
    
    // Skip dead particles
    if (p.life <= 0.0) return;
    
    // Apply gravity
    p.acceleration.y = gravity;
    
    // Apply drag
    p.velocity *= (1.0 - drag * delta_time);
    
    // Integrate velocity
    p.velocity += p.acceleration * delta_time;
    
    // Integrate position
    p.position += p.velocity * delta_time;
    
    // Integrate rotation
    p.rotation += p.angular_velocity * delta_time;
    
    // Update life
    p.life -= delta_time / p.max_life;
    
    // Update scale based on life
    float life_t = 1.0 - p.life;
    p.scale = mix(scale_start, scale_end, life_t);
    
    // Write back
    particles[idx] = p;
}
