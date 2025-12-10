// utils/aspect-ratio.js - Smart Aspect Ratio Handling
// Based on professional frontend architecture

export class AspectRatioManager {
    static RATIOS = {
        '9:16': { width: 1080, height: 1920, name: 'TikTok Vertical' },
        '1:1': { width: 1080, height: 1080, name: 'Square' },
        '16:9': { width: 1920, height: 1080, name: 'Widescreen' },
        '4:5': { width: 1080, height: 1350, name: 'Instagram Portrait' },
        '4:3': { width: 1440, height: 1080, name: 'Standard' },
        '3:4': { width: 1080, height: 1440, name: 'Portrait Standard' },
        '21:9': { width: 2560, height: 1080, name: 'Cinematic Ultrawide' },
        '2.35:1': { width: 1920, height: 817, name: 'Cinematic' }
    };

    /**
     * Parse aspect ratio string to numeric value
     * @param {string} ratioStr - e.g., "16:9", "1:1"
     * @returns {number|null} - Numeric aspect ratio (width/height)
     */
    static parseRatio(ratioStr) {
        if (!ratioStr || ratioStr === 'Custom') return null;
        const parts = ratioStr.split(':');
        if (parts.length === 2) {
            const w = parseFloat(parts[0]);
            const h = parseFloat(parts[1]);
            if (w > 0 && h > 0) {
                return w / h;
            }
        }
        return null;
    }

    /**
     * Get resolution for a given aspect ratio
     * @param {string} ratioStr - e.g., "16:9"
     * @returns {{width: number, height: number}}
     */
    static getResolution(ratioStr) {
        return this.RATIOS[ratioStr] || { width: 1920, height: 1080 };
    }

    /**
     * Calculate viewport dimensions for a container with target aspect ratio
     * @param {number} containerWidth - Container width in pixels
     * @param {number} containerHeight - Container height in pixels (can be 0 for auto)
     * @param {string} targetRatioStr - Target ratio string, e.g., "1:1"
     * @param {string} mode - 'contain', 'cover', or 'exact'
     * @returns {{width: number, height: number, offsetX: number, offsetY: number, scale: number}}
     */
    static calculateViewport(containerWidth, containerHeight, targetRatioStr, mode = 'contain') {
        const targetRatioConfig = this.RATIOS[targetRatioStr];
        if (!targetRatioConfig) {
            return { width: containerWidth, height: containerWidth, offsetX: 0, offsetY: 0, scale: 1 };
        }

        const targetAspect = targetRatioConfig.width / targetRatioConfig.height;

        let width, height, offsetX = 0, offsetY = 0;

        if (mode === 'exact') {
            // EXACT: Force exact aspect ratio, container height follows width
            width = containerWidth;
            height = containerWidth / targetAspect;
            offsetX = 0;
            offsetY = 0;
        } else if (mode === 'contain' && containerHeight > 0) {
            // CONTAIN: Fit entire content within viewport (letterbox)
            const containerAspect = containerWidth / containerHeight;

            if (containerAspect > targetAspect) {
                // Container wider than target - letterbox sides
                height = containerHeight;
                width = height * targetAspect;
                offsetX = (containerWidth - width) / 2;
                offsetY = 0;
            } else {
                // Container taller than target - letterbox top/bottom
                width = containerWidth;
                height = width / targetAspect;
                offsetX = 0;
                offsetY = (containerHeight - height) / 2;
            }
        } else if (mode === 'cover' && containerHeight > 0) {
            // COVER: Cover entire viewport (crop)
            const containerAspect = containerWidth / containerHeight;

            if (containerAspect > targetAspect) {
                width = containerWidth;
                height = width / targetAspect;
                offsetX = 0;
                offsetY = (containerHeight - height) / 2;
            } else {
                height = containerHeight;
                width = height * targetAspect;
                offsetX = (containerWidth - width) / 2;
                offsetY = 0;
            }
        } else {
            // Default: Calculate height from width using aspect ratio
            width = containerWidth;
            height = containerWidth / targetAspect;
        }

        const scale = width / targetRatioConfig.width;

        return {
            width: Math.round(width * 100) / 100,
            height: Math.round(height * 100) / 100,
            offsetX: Math.round(offsetX * 100) / 100,
            offsetY: Math.round(offsetY * 100) / 100,
            scale,
            pixelRatio: window.devicePixelRatio || 1,
            baseWidth: targetRatioConfig.width,
            baseHeight: targetRatioConfig.height
        };
    }

    /**
     * Get CSS aspect-ratio string
     * @param {string} ratioStr - e.g., "16:9"
     * @returns {string} - CSS format, e.g., "16 / 9"
     */
    static toCssRatio(ratioStr) {
        const parts = ratioStr?.split(':');
        if (parts && parts.length === 2) {
            return `${parts[0]} / ${parts[1]}`;
        }
        return '1 / 1';
    }

    /**
     * Get safe area guides for a given aspect ratio
     * @param {string} aspectRatio - e.g., "9:16"
     * @returns {{title: object, caption: object, action: object}}
     */
    static getSafeArea(aspectRatio) {
        const safeAreas = {
            '9:16': {
                title: { top: 0.1, bottom: 0.9 },
                caption: { top: 0.7, bottom: 0.95 },
                action: { left: 0.05, right: 0.95 }
            },
            '1:1': {
                title: { top: 0.15, bottom: 0.85 },
                caption: { top: 0.75, bottom: 0.95 },
                action: { left: 0.05, right: 0.95 }
            },
            '16:9': {
                title: { top: 0.1, bottom: 0.9 },
                caption: { top: 0.8, bottom: 0.95 },
                action: { left: 0.05, right: 0.95 }
            }
        };

        return safeAreas[aspectRatio] || safeAreas['9:16'];
    }
}

export default AspectRatioManager;
