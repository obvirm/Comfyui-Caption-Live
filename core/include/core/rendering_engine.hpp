/**
 * @file rendering_engine.hpp
 * @brief Modern C++ rendering engine interface with concepts and coroutines
 */

#pragma once

#include <memory>
#include <future>
#include <expected>
#include <concepts>
#include <span>
#include <string_view>
#include <cstdint>
#include <vector>

namespace CaptionEngine {

// Forward declarations
struct Template;
struct FrameData;
struct FrameContext;

/// Quality levels for adaptive rendering
enum class QualityLevel {
    Draft,      ///< Fast, for UI preview
    Preview,    ///< Balance, for timeline scrubbing
    Final       ///< Maximum, for final render
};

/// Compute backend types
enum class BackendType {
    Auto,       ///< Auto-detect best backend
    CPU,        ///< CPU fallback
    CUDA,       ///< NVIDIA CUDA
    Vulkan,     ///< Vulkan compute
    Metal,      ///< Apple Metal
    WebGPU,     ///< Browser WebGPU
    DirectX12   ///< Windows DirectX 12
};

/// Render error types
enum class RenderError {
    None,
    InvalidTemplate,
    OutOfMemory,
    BackendNotAvailable,
    ShaderCompilationFailed,
    FrameValidationFailed,
    Timeout
};

/// Concept for renderable objects
template<typename T>
concept Renderable = requires(T t, FrameContext& ctx) {
    { t.render(ctx) } -> std::same_as<void>;
    { t.get_bounds() } -> std::same_as<std::pair<float, float>>;
};

/// Frame result with metadata
struct FrameResult {
    std::vector<uint8_t> pixels;
    uint32_t width;
    uint32_t height;
    uint64_t checksum;
    double render_time_ms;
    QualityLevel quality;
    
    bool validate(const FrameResult& other) const {
        return checksum == other.checksum;
    }
};

/// Frame hash for consistency validation
struct FrameHash {
    uint64_t data_hash;
    uint64_t metadata_hash;
    uint32_t checksum;
    
    bool operator==(const FrameHash&) const = default;
};

/**
 * @brief Modern C++ Rendering Engine
 * 
 * Thread-safe, move-only engine with async rendering support.
 */
class RenderEngine {
public:
    /// Configuration for engine initialization
    struct Config {
        uint32_t width = 1920;
        uint32_t height = 1080;
        QualityLevel default_quality = QualityLevel::Final;
        BackendType preferred_backend = BackendType::Auto;
        size_t max_memory_mb = 512;
        bool enable_validation = true;
    };
    
    /// Create engine with configuration
    explicit RenderEngine(const Config& config);
    
    /// Destructor
    ~RenderEngine();
    
    /// Move-only semantics
    RenderEngine(RenderEngine&&) noexcept;
    RenderEngine& operator=(RenderEngine&&) noexcept;
    
    /// No copying
    RenderEngine(const RenderEngine&) = delete;
    RenderEngine& operator=(const RenderEngine&) = delete;
    
    /// Synchronous rendering
    [[nodiscard]] std::expected<FrameResult, RenderError> 
    render_frame(const Template& tmpl, double time, QualityLevel quality = QualityLevel::Final);
    
    /// Asynchronous rendering with future
    [[nodiscard]] std::future<FrameResult> 
    render_frame_async(const Template& tmpl, double time, QualityLevel quality = QualityLevel::Final);
    
    /// Deterministic rendering with hash validation
    [[nodiscard]] std::expected<FrameResult, RenderError>
    render_frame_deterministic(const Template& tmpl, uint64_t frame_number, const FrameHash& expected_hash);
    
    /// Get current backend type
    [[nodiscard]] BackendType current_backend() const noexcept;
    
    /// Get available backends
    [[nodiscard]] std::vector<BackendType> available_backends() const;
    
    /// Compute frame hash
    [[nodiscard]] FrameHash compute_hash(const FrameResult& frame) const;
    
    /// Resize output dimensions
    void resize(uint32_t width, uint32_t height);
    
    /// Set quality level
    void set_quality(QualityLevel quality);
    
    /// Clear cached resources
    void clear_cache();
    
    /// Get memory usage in bytes
    [[nodiscard]] size_t memory_usage() const noexcept;

private:
    /// PIMPL idiom for ABI stability
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief Abstract compute backend interface
 */
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    
    /// Get backend type
    [[nodiscard]] virtual BackendType type() const noexcept = 0;
    
    /// Check if backend is available
    [[nodiscard]] virtual bool is_available() const noexcept = 0;
    
    /// Initialize backend
    virtual void initialize() = 0;
    
    /// Dispatch compute kernel
    virtual void dispatch_compute(
        std::string_view kernel_name,
        std::span<const uint8_t> params,
        uint32_t work_groups_x,
        uint32_t work_groups_y,
        uint32_t work_groups_z
    ) = 0;
    
    /// Create GPU buffer
    [[nodiscard]] virtual uint64_t create_buffer(size_t size, bool host_visible) = 0;
    
    /// Destroy buffer
    virtual void destroy_buffer(uint64_t handle) = 0;
    
    /// Upload data to buffer
    virtual void upload_buffer(uint64_t handle, std::span<const uint8_t> data) = 0;
    
    /// Download data from buffer
    virtual void download_buffer(uint64_t handle, std::span<uint8_t> data) = 0;
    
    /// Synchronize all pending operations
    virtual void synchronize() = 0;
    
    /// Factory method
    [[nodiscard]] static std::unique_ptr<ComputeBackend> create(BackendType type);
};

/**
 * @brief Effect graph for complex effect chains
 */
class EffectGraph {
public:
    struct Node {
        std::string effect_type;
        std::vector<std::pair<std::string, std::string>> parameters;
        std::vector<size_t> inputs;
    };
    
    /// Add node to graph
    size_t add_node(Node node);
    
    /// Connect nodes
    void connect(size_t from, size_t to);
    
    /// Compile graph for execution
    [[nodiscard]] bool compile();
    
    /// Execute graph
    void execute(FrameContext& ctx);
    
private:
    std::vector<Node> nodes_;
    std::vector<std::pair<size_t, size_t>> edges_;
    bool compiled_ = false;
};

} // namespace CaptionEngine
