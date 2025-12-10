/**
 * Caption Engine - JavaScript wrapper for WASM
 */

let wasmModule = null;
let isInitialized = false;

/**
 * Load and initialize the Caption Engine WASM module
 */
export async function loadCaptionEngine() {
    if (wasmModule) {
        return wasmModule;
    }

    try {
        // Try to load WASM module
        const Module = await import('./caption_engine.js');
        wasmModule = await Module.default();
        isInitialized = true;

        return {
            init: (width, height) => {
                wasmModule.initEngine(width, height);
            },

            renderFrame: (templateJson, time) => {
                return wasmModule.renderFrame(templateJson, time);
            },

            renderToImageData: (templateJson, time) => {
                return wasmModule.renderToImageData(templateJson, time);
            },

            computeHash: (templateJson, time) => {
                return wasmModule.computeHash(templateJson, time);
            },

            getBackend: () => {
                return wasmModule.getBackend();
            },

            isWebGPUAvailable: () => {
                return wasmModule.isWebGPUAvailable();
            }
        };
    } catch (error) {
        console.error('Failed to load Caption Engine WASM:', error);
        throw error;
    }
}

/**
 * Create a template from simple parameters
 */
export function createTemplate(options = {}) {
    const {
        text = '',
        animation = 'none',
        style = {},
        position = {},
        canvas = {},
        duration = 5.0,
        fps = 60.0
    } = options;

    const template = {
        canvas: {
            width: canvas.width || 1920,
            height: canvas.height || 1080
        },
        duration,
        fps,
        layers: [{
            type: 'text',
            content: text,
            position: {
                x: position.x || 0.5,
                y: position.y || 0.5
            },
            style: {
                font_family: style.font_family || 'Inter',
                font_size: style.font_size || 50,
                font_weight: style.font_weight || 900,
                color: style.color || '#FFFFFF',
                stroke_color: style.stroke_color,
                stroke_width: style.stroke_width || 0
            }
        }]
    };

    // Add animation if specified
    if (animation !== 'none') {
        template.layers[0].animation = {
            type: animation,
            segments: [{
                text: text,
                start: 0,
                end: duration
            }]
        };
    }

    return template;
}

/**
 * Validate a template JSON string
 */
export function validateTemplate(json) {
    const errors = [];

    try {
        const template = JSON.parse(json);

        // Validate canvas
        if (!template.canvas) {
            errors.push('Missing canvas property');
        } else {
            if (!template.canvas.width || template.canvas.width <= 0) {
                errors.push('Invalid canvas width');
            }
            if (!template.canvas.height || template.canvas.height <= 0) {
                errors.push('Invalid canvas height');
            }
        }

        // Validate duration
        if (!template.duration || template.duration <= 0) {
            errors.push('Invalid or missing duration');
        }

        // Validate fps
        if (!template.fps || template.fps <= 0) {
            errors.push('Invalid or missing fps');
        }

        // Validate layers
        if (!template.layers || !Array.isArray(template.layers)) {
            errors.push('Missing or invalid layers array');
        } else {
            template.layers.forEach((layer, i) => {
                if (!layer.type) {
                    errors.push(`Layer ${i}: missing type`);
                }
                if (layer.type === 'text' && !layer.content && layer.content !== '') {
                    errors.push(`Layer ${i}: text layer missing content`);
                }
                if (layer.type === 'image' && !layer.src) {
                    errors.push(`Layer ${i}: image layer missing src`);
                }
            });
        }

    } catch (e) {
        errors.push(`JSON parse error: ${e.message}`);
    }

    return {
        valid: errors.length === 0,
        errors
    };
}

// Export Quality enum
export const Quality = {
    DRAFT: 'draft',
    PREVIEW: 'preview',
    FINAL: 'final'
};
