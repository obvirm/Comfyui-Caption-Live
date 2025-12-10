// preview-optimizer.js - Adaptive Quality System
// Automatically adjusts preview quality based on performance

/**
 * PreviewOptimizer monitors performance and adjusts preview quality
 * to maintain smooth playback across different hardware.
 */
export class PreviewOptimizer {
    constructor(renderer = null) {
        this.renderer = renderer;

        // Quality level configurations
        this.qualityLevels = {
            low: {
                resolution: 0.5,      // 50% of target resolution
                samples: 1,           // No multisampling
                antialias: false,
                particles: 100,
                updateRate: 30,       // 30 fps
                label: 'Low (Fast)'
            },
            medium: {
                resolution: 0.75,     // 75% of target resolution
                samples: 2,
                antialias: true,
                particles: 1000,
                updateRate: 60,       // 60 fps
                label: 'Medium'
            },
            high: {
                resolution: 1.0,      // Full resolution
                samples: 4,
                antialias: true,
                particles: 10000,
                updateRate: 60,
                label: 'High'
            }
        };

        // Current state
        this.currentQuality = 'medium';
        this.adaptiveEnabled = true;

        // Performance monitoring
        this.fpsHistory = [];
        this.memoryHistory = [];
        this.historySize = 60; // ~1 second at 60fps

        // Thresholds
        this.fpsDropThreshold = 24;
        this.fpsRaiseThreshold = 50;
        this.memoryThreshold = 0.7; // 70% memory usage

        // Cooldown to prevent rapid switching
        this.lastQualityChange = 0;
        this.qualityChangeCooldown = 2000; // 2 seconds

        // Event callbacks
        this.onQualityChange = null;
    }

    /**
     * Record frame time for FPS calculation
     * @param {number} frameTime - Time in ms for this frame
     */
    recordFrame(frameTime) {
        const fps = 1000 / frameTime;

        this.fpsHistory.push(fps);
        if (this.fpsHistory.length > this.historySize) {
            this.fpsHistory.shift();
        }

        // Check if we need to adapt quality
        if (this.adaptiveEnabled) {
            this.adaptQualityBasedOnPerformance();
        }
    }

    /**
     * Get current average FPS
     */
    getCurrentFPS() {
        if (this.fpsHistory.length === 0) return 60;
        const sum = this.fpsHistory.reduce((a, b) => a + b, 0);
        return sum / this.fpsHistory.length;
    }

    /**
     * Get estimated memory usage (0-1)
     */
    getMemoryUsage() {
        if (performance.memory) {
            const used = performance.memory.usedJSHeapSize;
            const total = performance.memory.jsHeapSizeLimit;
            return used / total;
        }
        return 0.5; // Default if API not available
    }

    /**
     * Adapt quality based on current performance
     */
    adaptQualityBasedOnPerformance() {
        const now = Date.now();

        // Respect cooldown
        if (now - this.lastQualityChange < this.qualityChangeCooldown) {
            return;
        }

        const fps = this.getCurrentFPS();
        const memory = this.getMemoryUsage();

        let newQuality = this.currentQuality;

        // Drop quality if struggling
        if (fps < this.fpsDropThreshold || memory > this.memoryThreshold) {
            if (this.currentQuality === 'high') {
                newQuality = 'medium';
            } else if (this.currentQuality === 'medium') {
                newQuality = 'low';
            }
        }
        // Raise quality if we have headroom
        else if (fps > this.fpsRaiseThreshold && memory < this.memoryThreshold - 0.1) {
            if (this.currentQuality === 'low') {
                newQuality = 'medium';
            } else if (this.currentQuality === 'medium') {
                newQuality = 'high';
            }
        }

        if (newQuality !== this.currentQuality) {
            this.setQuality(newQuality);
        }
    }

    /**
     * Set quality level
     * @param {string} level - 'low', 'medium', or 'high'
     */
    setQuality(level) {
        if (!this.qualityLevels[level]) {
            console.warn(`Unknown quality level: ${level}`);
            return;
        }

        const prevQuality = this.currentQuality;
        this.currentQuality = level;
        this.lastQualityChange = Date.now();

        const settings = this.qualityLevels[level];

        // Apply settings to renderer if available
        if (this.renderer) {
            if (this.renderer.setResolutionScale) {
                this.renderer.setResolutionScale(settings.resolution);
            }
            if (this.renderer.setAntialiasing) {
                this.renderer.setAntialiasing(settings.antialias);
            }
            if (this.renderer.setMaxParticles) {
                this.renderer.setMaxParticles(settings.particles);
            }
        }

        console.log(`[PreviewOptimizer] Quality: ${prevQuality} → ${level} (FPS: ${this.getCurrentFPS().toFixed(1)})`);

        // Fire callback
        if (this.onQualityChange) {
            this.onQualityChange(level, settings);
        }
    }

    /**
     * Force a specific quality level (disables adaptive)
     */
    forceQuality(level) {
        this.adaptiveEnabled = false;
        this.setQuality(level);
    }

    /**
     * Re-enable adaptive quality
     */
    enableAdaptive() {
        this.adaptiveEnabled = true;
    }

    /**
     * Get current quality settings
     */
    getCurrentSettings() {
        return this.qualityLevels[this.currentQuality];
    }

    /**
     * Get quality indicator for UI display
     */
    getQualityIndicator() {
        const settings = this.qualityLevels[this.currentQuality];
        return {
            level: this.currentQuality,
            label: settings.label,
            fps: Math.round(this.getCurrentFPS()),
            memory: Math.round(this.getMemoryUsage() * 100),
            adaptive: this.adaptiveEnabled
        };
    }

    /**
     * Reset performance history
     */
    reset() {
        this.fpsHistory = [];
        this.memoryHistory = [];
        this.currentQuality = 'medium';
        this.lastQualityChange = 0;
    }

    /**
     * Create quality indicator DOM element
     */
    createIndicatorElement() {
        const container = document.createElement('div');
        container.className = 'quality-indicator';
        container.style.cssText = `
      position: absolute;
      bottom: 5px;
      left: 5px;
      background: rgba(0, 0, 0, 0.6);
      color: white;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-family: monospace;
      z-index: 100;
      pointer-events: none;
    `;

        const updateIndicator = () => {
            const info = this.getQualityIndicator();
            container.textContent = `${info.fps}fps | ${info.label}`;
            container.style.color = info.fps < 30 ? '#ff6b6b' :
                info.fps < 50 ? '#ffd93d' : '#6bff6b';
        };

        // Update every second
        setInterval(updateIndicator, 1000);
        updateIndicator();

        return container;
    }
}

export default PreviewOptimizer;
