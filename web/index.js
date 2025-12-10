// index.js - Caption Live Frontend Architecture
// Export all modules for unified access

// Core utilities
export { AspectRatioManager } from './aspect-ratio.js';
export { VirtualCamera } from './virtual-camera.js';

// Preview system
export { PreviewOptimizer } from './preview-optimizer.js';
export { SafeAreaOverlay } from './safe-area-overlay.js';

// Layout system
export { setupLayout, setupWidgetCallbacks } from './layout.js';

// UI components
export { createPreviewUI } from './ui.js';

// Renderer
export { startRenderLoop } from './renderer.js';

/**
 * Initialize complete preview system with all features
 * @param {Object} options - Configuration options
 * @returns {Object} - Initialized preview system
 */
export function initPreviewSystem(options = {}) {
    const {
        canvas = null,
        aspectRatio = '16:9',
        quality = 'medium',
        showSafeArea = false,
        adaptiveQuality = true
    } = options;

    // Create virtual camera
    const camera = VirtualCamera.fromAspectRatio(aspectRatio);

    // Create preview optimizer
    const optimizer = new PreviewOptimizer();
    optimizer.adaptiveEnabled = adaptiveQuality;
    if (quality !== 'auto') {
        optimizer.forceQuality(quality);
    }

    // Create safe area overlay (if canvas provided)
    let safeArea = null;
    if (canvas) {
        safeArea = new SafeAreaOverlay(canvas, aspectRatio);
        if (showSafeArea) {
            safeArea.show();
        }
    }

    return {
        camera,
        optimizer,
        safeArea,

        // Convenience methods
        setAspectRatio(ratio) {
            camera.updateConfig(AspectRatioManager.getResolution(ratio));
            if (safeArea) {
                safeArea.setAspectRatio(ratio);
            }
        },

        setQuality(level) {
            optimizer.setQuality(level);
        },

        toggleSafeArea() {
            if (safeArea) {
                safeArea.toggle();
            }
        },

        destroy() {
            if (safeArea) {
                safeArea.destroy();
            }
        }
    };
}

// Version info
export const VERSION = '2.0.0';
export const ARCHITECTURE = 'Professional Frontend Preview';
