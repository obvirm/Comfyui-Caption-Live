// safe-area-overlay.js - Visual Guides for Content Safe Areas
// Shows safe zones for different platforms (TikTok, Reels, etc.)

import { AspectRatioManager } from './aspect-ratio.js';

/**
 * SafeAreaOverlay provides visual guides for content creators
 * to ensure their content is visible on all platforms.
 */
export class SafeAreaOverlay {
    constructor(canvas, aspectRatio = '9:16') {
        this.canvas = canvas;
        this.aspectRatio = aspectRatio;
        this.visible = false;
        this.overlayCanvas = null;
        this.ctx = null;

        // Guide colors
        this.colors = {
            title: 'rgba(255, 100, 100, 0.5)',     // Red for title safe zone
            caption: 'rgba(100, 255, 100, 0.5)',   // Green for caption area
            action: 'rgba(100, 100, 255, 0.5)',    // Blue for interactive elements
            grid: 'rgba(255, 255, 255, 0.2)',      // White for rule of thirds
            center: 'rgba(255, 255, 0, 0.5)'       // Yellow for center markers
        };

        // Safe area definitions per platform
        this.platformSafeAreas = {
            tiktok: {
                '9:16': {
                    titleTop: 0.08,      // Username, share button area
                    titleBottom: 0.85,   // Caption, music info area
                    actionLeft: 0.03,    // Left edge safe
                    actionRight: 0.85    // Right side icons
                }
            },
            instagram: {
                '9:16': {
                    titleTop: 0.05,
                    titleBottom: 0.88,
                    actionLeft: 0.03,
                    actionRight: 0.90
                },
                '4:5': {
                    titleTop: 0.05,
                    titleBottom: 0.85,
                    actionLeft: 0.03,
                    actionRight: 0.97
                }
            },
            youtube: {
                '16:9': {
                    titleTop: 0.05,
                    titleBottom: 0.90,
                    actionLeft: 0.05,
                    actionRight: 0.95
                }
            }
        };

        this.currentPlatform = 'tiktok';
    }

    /**
     * Initialize overlay canvas
     */
    init() {
        if (!this.overlayCanvas) {
            this.overlayCanvas = document.createElement('canvas');
            this.overlayCanvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 50;
      `;
            this.ctx = this.overlayCanvas.getContext('2d');

            if (this.canvas.parentNode) {
                this.canvas.parentNode.appendChild(this.overlayCanvas);
            }
        }

        this.resize();
    }

    /**
     * Resize overlay to match canvas
     */
    resize() {
        if (!this.overlayCanvas) return;

        this.overlayCanvas.width = this.canvas.width;
        this.overlayCanvas.height = this.canvas.height;

        if (this.visible) {
            this.draw();
        }
    }

    /**
     * Show safe area overlay
     */
    show() {
        if (!this.overlayCanvas) {
            this.init();
        }
        this.visible = true;
        this.overlayCanvas.style.display = 'block';
        this.draw();
    }

    /**
     * Hide safe area overlay
     */
    hide() {
        this.visible = false;
        if (this.overlayCanvas) {
            this.overlayCanvas.style.display = 'none';
        }
    }

    /**
     * Toggle visibility
     */
    toggle() {
        if (this.visible) {
            this.hide();
        } else {
            this.show();
        }
    }

    /**
     * Set aspect ratio
     */
    setAspectRatio(ratio) {
        this.aspectRatio = ratio;
        if (this.visible) {
            this.draw();
        }
    }

    /**
     * Set platform for platform-specific safe areas
     */
    setPlatform(platform) {
        this.currentPlatform = platform;
        if (this.visible) {
            this.draw();
        }
    }

    /**
     * Draw all safe area guides
     */
    draw() {
        if (!this.ctx) return;

        const { width, height } = this.overlayCanvas;

        // Clear canvas
        this.ctx.clearRect(0, 0, width, height);

        // Get safe area for current platform and aspect ratio
        const safeArea = this.getSafeArea();

        // Draw safe zones
        this.drawSafeZones(safeArea, width, height);

        // Draw rule of thirds grid
        this.drawRuleOfThirds(width, height);

        // Draw center markers
        this.drawCenterMarkers(width, height);
    }

    /**
     * Get safe area config for current settings
     */
    getSafeArea() {
        const platformAreas = this.platformSafeAreas[this.currentPlatform];
        if (platformAreas && platformAreas[this.aspectRatio]) {
            return platformAreas[this.aspectRatio];
        }

        // Fallback to generic safe area from AspectRatioManager
        const genericArea = AspectRatioManager.getSafeArea(this.aspectRatio);
        return {
            titleTop: genericArea.title.top,
            titleBottom: genericArea.title.bottom,
            actionLeft: genericArea.action?.left || 0.05,
            actionRight: genericArea.action?.right || 0.95
        };
    }

    /**
     * Draw safe zones (title, caption, action areas)
     */
    drawSafeZones(safeArea, width, height) {
        // Title danger zone (top)
        this.ctx.fillStyle = this.colors.title;
        this.ctx.fillRect(0, 0, width, height * safeArea.titleTop);

        // Caption danger zone (bottom)
        this.ctx.fillStyle = this.colors.caption;
        this.ctx.fillRect(0, height * safeArea.titleBottom, width, height * (1 - safeArea.titleBottom));

        // Action danger zone (right side on TikTok-style apps)
        if (safeArea.actionRight < 1) {
            this.ctx.fillStyle = this.colors.action;
            this.ctx.fillRect(width * safeArea.actionRight, 0, width * (1 - safeArea.actionRight), height);
        }

        // Draw safe area border
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(
            width * safeArea.actionLeft,
            height * safeArea.titleTop,
            width * (safeArea.actionRight - safeArea.actionLeft),
            height * (safeArea.titleBottom - safeArea.titleTop)
        );

        // Draw label
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            'Safe Area',
            width / 2,
            height * safeArea.titleTop + 20
        );
    }

    /**
     * Draw rule of thirds grid
     */
    drawRuleOfThirds(width, height) {
        this.ctx.strokeStyle = this.colors.grid;
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([5, 5]);

        // Vertical lines
        for (let i = 1; i < 3; i++) {
            const x = (width / 3) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.stroke();
        }

        // Horizontal lines
        for (let i = 1; i < 3; i++) {
            const y = (height / 3) * i;
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            this.ctx.stroke();
        }

        this.ctx.setLineDash([]);
    }

    /**
     * Draw center markers
     */
    drawCenterMarkers(width, height) {
        const cx = width / 2;
        const cy = height / 2;
        const size = 20;

        this.ctx.strokeStyle = this.colors.center;
        this.ctx.lineWidth = 2;

        // Horizontal line
        this.ctx.beginPath();
        this.ctx.moveTo(cx - size, cy);
        this.ctx.lineTo(cx + size, cy);
        this.ctx.stroke();

        // Vertical line
        this.ctx.beginPath();
        this.ctx.moveTo(cx, cy - size);
        this.ctx.lineTo(cx, cy + size);
        this.ctx.stroke();

        // Center point
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        this.ctx.fillStyle = this.colors.center;
        this.ctx.fill();
    }

    /**
     * Destroy overlay
     */
    destroy() {
        if (this.overlayCanvas && this.overlayCanvas.parentNode) {
            this.overlayCanvas.parentNode.removeChild(this.overlayCanvas);
        }
        this.overlayCanvas = null;
        this.ctx = null;
    }
}

export default SafeAreaOverlay;
