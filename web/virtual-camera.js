// virtual-camera.js - Unified Coordinate System
// Ensures consistent rendering across browser preview and backend output

import { AspectRatioManager } from './aspect-ratio.js';

/**
 * VirtualCamera provides a unified coordinate system for rendering
 * that works identically in browser preview and backend output.
 */
export class VirtualCamera {
    constructor(config = {}) {
        // STANDARD: 1080x1920 (9:16) as base for TikTok
        this.baseWidth = config.baseWidth || 1080;
        this.baseHeight = config.baseHeight || 1920;
        this.baseAspect = this.baseWidth / this.baseHeight;

        // User customization
        this.targetWidth = config.width || 1080;
        this.targetHeight = config.height || 1920;
        this.targetAspect = this.targetWidth / this.targetHeight;

        // Viewport mode: 'contain', 'cover', 'stretch'
        this.viewportMode = config.viewportMode || 'contain';

        // Camera transform matrix for consistent rendering
        this.matrix = this.calculateProjectionMatrix();
    }

    /**
     * Create VirtualCamera from aspect ratio string
     * @param {string} ratioStr - e.g., "1:1", "16:9"
     * @returns {VirtualCamera}
     */
    static fromAspectRatio(ratioStr) {
        const resolution = AspectRatioManager.getResolution(ratioStr);
        return new VirtualCamera({
            width: resolution.width,
            height: resolution.height,
            baseWidth: resolution.width,
            baseHeight: resolution.height
        });
    }

    /**
     * Convert screen coordinates to virtual coordinates
     * @param {number} x - Screen X position
     * @param {number} y - Screen Y position
     * @param {number} viewportWidth - Viewport width in pixels
     * @param {number} viewportHeight - Viewport height in pixels
     * @returns {{x: number, y: number}} - Virtual coordinates
     */
    screenToVirtual(x, y, viewportWidth, viewportHeight) {
        // Normalize to [0, 1]
        const nx = x / viewportWidth;
        const ny = y / viewportHeight;

        // Apply viewport fitting
        const fitted = this.fitToViewport(nx, ny, viewportWidth, viewportHeight);

        // Scale to virtual resolution
        return {
            x: fitted.x * this.baseWidth,
            y: fitted.y * this.baseHeight
        };
    }

    /**
     * Convert virtual coordinates to screen coordinates
     * @param {number} vx - Virtual X position (0-baseWidth)
     * @param {number} vy - Virtual Y position (0-baseHeight)
     * @param {number} viewportWidth - Viewport width in pixels
     * @param {number} viewportHeight - Viewport height in pixels
     * @returns {{x: number, y: number}} - Screen coordinates
     */
    virtualToScreen(vx, vy, viewportWidth, viewportHeight) {
        // Normalize to [0, 1]
        const nx = vx / this.baseWidth;
        const ny = vy / this.baseHeight;

        // Apply viewport transform
        const viewport = this.getViewportTransform(viewportWidth, viewportHeight);

        return {
            x: viewport.offsetX + nx * viewport.width,
            y: viewport.offsetY + ny * viewport.height
        };
    }

    /**
     * Fit normalized coordinates to viewport
     */
    fitToViewport(nx, ny, viewportWidth, viewportHeight) {
        const viewport = this.getViewportTransform(viewportWidth, viewportHeight);

        // Convert screen position to content position
        const contentX = (nx * viewportWidth - viewport.offsetX) / viewport.width;
        const contentY = (ny * viewportHeight - viewport.offsetY) / viewport.height;

        return {
            x: Math.max(0, Math.min(1, contentX)),
            y: Math.max(0, Math.min(1, contentY))
        };
    }

    /**
     * Get viewport transform for current configuration
     */
    getViewportTransform(viewportWidth, viewportHeight) {
        const viewportAspect = viewportWidth / viewportHeight;

        let width, height, offsetX, offsetY;

        if (this.viewportMode === 'contain') {
            if (viewportAspect > this.targetAspect) {
                // Viewport wider - letterbox sides
                height = viewportHeight;
                width = height * this.targetAspect;
                offsetX = (viewportWidth - width) / 2;
                offsetY = 0;
            } else {
                // Viewport taller - letterbox top/bottom
                width = viewportWidth;
                height = width / this.targetAspect;
                offsetX = 0;
                offsetY = (viewportHeight - height) / 2;
            }
        } else if (this.viewportMode === 'cover') {
            if (viewportAspect > this.targetAspect) {
                width = viewportWidth;
                height = width / this.targetAspect;
                offsetX = 0;
                offsetY = (viewportHeight - height) / 2;
            } else {
                height = viewportHeight;
                width = height * this.targetAspect;
                offsetX = (viewportWidth - width) / 2;
                offsetY = 0;
            }
        } else {
            // Stretch
            width = viewportWidth;
            height = viewportHeight;
            offsetX = 0;
            offsetY = 0;
        }

        return {
            width,
            height,
            offsetX,
            offsetY,
            scale: width / this.targetWidth
        };
    }

    /**
     * Calculate orthographic projection matrix for 2D rendering
     * Compatible with WebGL/WebGPU
     * @returns {Float32Array} - 4x4 projection matrix
     */
    calculateProjectionMatrix() {
        // Orthographic projection for 2D
        // Maps (0, 0) to (-1, 1) and (width, height) to (1, -1) in NDC
        const left = 0;
        const right = this.baseWidth;
        const bottom = this.baseHeight;
        const top = 0;
        const near = -1;
        const far = 1;

        return new Float32Array([
            2 / (right - left), 0, 0, 0,
            0, 2 / (top - bottom), 0, 0,
            0, 0, -2 / (far - near), 0,
            -(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            -(far + near) / (far - near),
            1
        ]);
    }

    /**
     * Get scale factor for font/element sizing
     * Based on reference 1080p height
     * @returns {number}
     */
    getScaleFactor() {
        // Scale relative to 1080p height
        // 16:9 (1080h) -> 1.0x
        // 9:16 (1920h) -> 1.77x (Text grows for vertical video)
        return this.targetHeight / 1080.0;
    }

    /**
     * Convert normalized position (0-1) to pixel position
     */
    normalizedToPixel(nx, ny) {
        return {
            x: nx * this.targetWidth,
            y: ny * this.targetHeight
        };
    }

    /**
     * Convert pixel position to normalized (0-1)
     */
    pixelToNormalized(px, py) {
        return {
            x: px / this.targetWidth,
            y: py / this.targetHeight
        };
    }

    /**
     * Update camera configuration
     */
    updateConfig(config) {
        if (config.width) this.targetWidth = config.width;
        if (config.height) this.targetHeight = config.height;
        if (config.viewportMode) this.viewportMode = config.viewportMode;

        this.targetAspect = this.targetWidth / this.targetHeight;
        this.matrix = this.calculateProjectionMatrix();
    }

    /**
     * Clone camera with modifications
     */
    clone(overrides = {}) {
        return new VirtualCamera({
            width: overrides.width || this.targetWidth,
            height: overrides.height || this.targetHeight,
            baseWidth: overrides.baseWidth || this.baseWidth,
            baseHeight: overrides.baseHeight || this.baseHeight,
            viewportMode: overrides.viewportMode || this.viewportMode
        });
    }

    /**
     * Serialize camera state for sync with backend
     */
    toJSON() {
        return {
            baseWidth: this.baseWidth,
            baseHeight: this.baseHeight,
            targetWidth: this.targetWidth,
            targetHeight: this.targetHeight,
            viewportMode: this.viewportMode
        };
    }

    /**
     * Restore from serialized state
     */
    static fromJSON(json) {
        return new VirtualCamera({
            baseWidth: json.baseWidth,
            baseHeight: json.baseHeight,
            width: json.targetWidth,
            height: json.targetHeight,
            viewportMode: json.viewportMode
        });
    }
}

export default VirtualCamera;
