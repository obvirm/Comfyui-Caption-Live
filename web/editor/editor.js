// editor.js - Caption Live Editor JavaScript
// Handles all interactivity and state management

class CaptionEditor {
    constructor() {
        // Template state
        this.template = {
            caption_text: "Hello World",
            duration: 5.0,
            aspect_ratio: "1:1",
            style: "box",
            highlight_color: "#39E55F",
            text_color: "#FFFFFF",
            stroke_color: "#000000",
            stroke_width: 2,
            font_size: 24,
            pos_x: 0.5,
            pos_y: 0.8,
            fps: 60,
            quality: "Medium"
        };

        // Playback state
        this.isPlaying = false;
        this.currentTime = 0;
        this.animationFrame = null;
        this.lastFrameTime = 0;

        // UI state
        this.showSafeArea = false;
        this.showGrid = false;
        this.zoom = 100;

        // Initialize
        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.updatePreview();
        this.updateTimeline();
        this.updateStats();
    }

    bindElements() {
        // Header
        this.btnPlay = document.getElementById('btnPlay');
        this.currentTimeDisplay = document.getElementById('currentTime');
        this.totalTimeDisplay = document.getElementById('totalTime');
        this.fpsDisplay = document.getElementById('fpsDisplay');
        this.btnExport = document.getElementById('btnExport');

        // Text properties
        this.captionText = document.getElementById('captionText');
        this.stylePresets = document.getElementById('stylePresets');

        // Colors
        this.highlightColor = document.getElementById('highlightColor');
        this.highlightColorText = document.getElementById('highlightColorText');
        this.textColor = document.getElementById('textColor');
        this.textColorText = document.getElementById('textColorText');
        this.strokeColor = document.getElementById('strokeColor');
        this.strokeColorText = document.getElementById('strokeColorText');

        // Canvas
        this.captionPreview = document.getElementById('captionPreview');
        this.aspectContainer = document.getElementById('aspectContainer');
        this.safeAreaOverlay = document.getElementById('safeAreaOverlay');
        this.btnGrid = document.getElementById('btnGrid');
        this.btnSafeArea = document.getElementById('btnSafeArea');
        this.qualitySelect = document.getElementById('qualitySelect');

        // Settings sliders
        this.durationSlider = document.getElementById('durationSlider');
        this.durationValue = document.getElementById('durationValue');
        this.aspectSelect = document.getElementById('aspectSelect');
        this.aspectValue = document.getElementById('aspectValue');
        this.posXSlider = document.getElementById('posXSlider');
        this.posXValue = document.getElementById('posXValue');
        this.posYSlider = document.getElementById('posYSlider');
        this.posYValue = document.getElementById('posYValue');
        this.fontSizeSlider = document.getElementById('fontSizeSlider');
        this.fontSizeValue = document.getElementById('fontSizeValue');
        this.strokeWidthSlider = document.getElementById('strokeWidthSlider');
        this.strokeWidthValue = document.getElementById('strokeWidthValue');

        // Stats
        this.statDuration = document.getElementById('statDuration');
        this.statAspect = document.getElementById('statAspect');
        this.statSize = document.getElementById('statSize');

        // Timeline
        this.timeRuler = document.getElementById('timeRuler');
        this.playhead = document.getElementById('playhead');
        this.captionDuration = document.getElementById('captionDuration');
        this.btnZoomIn = document.getElementById('btnZoomIn');
        this.btnZoomOut = document.getElementById('btnZoomOut');
        this.zoomLevel = document.getElementById('zoomLevel');

        // Actions
        this.btnCopy = document.getElementById('btnCopy');
        this.btnReset = document.getElementById('btnReset');
        this.btnSavePreset = document.getElementById('btnSavePreset');
    }

    bindEvents() {
        // Play/Pause
        this.btnPlay.addEventListener('click', () => this.togglePlay());

        // Caption text
        this.captionText.addEventListener('input', (e) => {
            this.template.caption_text = e.target.value;
            this.updatePreview();
        });

        // Style presets
        this.stylePresets.querySelectorAll('.style-preset').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.stylePresets.querySelectorAll('.style-preset').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.template.style = e.target.dataset.style;
                this.updatePreview();
            });
        });

        // Color pickers
        this.bindColorPicker('highlight');
        this.bindColorPicker('text');
        this.bindColorPicker('stroke');

        // Sliders
        this.bindSlider('duration', (v) => {
            this.template.duration = v;
            this.durationValue.textContent = v.toFixed(1) + 's';
            this.updateTimeline();
            this.updateStats();
        });

        this.bindSlider('posX', (v) => {
            this.template.pos_x = v;
            this.posXValue.textContent = v.toFixed(2);
            this.updatePreview();
        });

        this.bindSlider('posY', (v) => {
            this.template.pos_y = v;
            this.posYValue.textContent = v.toFixed(2);
            this.updatePreview();
        });

        this.bindSlider('fontSize', (v) => {
            this.template.font_size = v;
            this.fontSizeValue.textContent = v + 'px';
            this.updatePreview();
        });

        this.bindSlider('strokeWidth', (v) => {
            this.template.stroke_width = v;
            this.strokeWidthValue.textContent = v + 'px';
            this.updatePreview();
        });

        // Aspect ratio
        this.aspectSelect.addEventListener('change', (e) => {
            this.template.aspect_ratio = e.target.value;
            this.aspectValue.textContent = e.target.value;
            this.updateAspectRatio();
            this.updateStats();
        });

        // Quality
        this.qualitySelect.addEventListener('change', (e) => {
            this.template.quality = e.target.value;
        });

        // Toggle buttons
        this.btnGrid.addEventListener('click', () => {
            this.showGrid = !this.showGrid;
            this.btnGrid.style.background = this.showGrid ? '#39e55f' : '#333';
        });

        this.btnSafeArea.addEventListener('click', () => {
            this.showSafeArea = !this.showSafeArea;
            this.safeAreaOverlay.style.display = this.showSafeArea ? 'block' : 'none';
            this.btnSafeArea.style.background = this.showSafeArea ? '#39e55f' : '#333';
        });

        // Timeline zoom
        this.btnZoomIn.addEventListener('click', () => {
            this.zoom = Math.min(200, this.zoom + 20);
            this.zoomLevel.textContent = this.zoom + '%';
        });

        this.btnZoomOut.addEventListener('click', () => {
            this.zoom = Math.max(50, this.zoom - 20);
            this.zoomLevel.textContent = this.zoom + '%';
        });

        // Actions
        this.btnCopy.addEventListener('click', () => this.copySettings());
        this.btnReset.addEventListener('click', () => this.resetToDefault());
        this.btnExport.addEventListener('click', () => this.exportToComfyUI());
    }

    bindColorPicker(name) {
        const colorInput = document.getElementById(name + 'Color');
        const textInput = document.getElementById(name + 'ColorText');
        const templateKey = name + '_color';

        colorInput.addEventListener('input', (e) => {
            this.template[templateKey] = e.target.value;
            textInput.value = e.target.value;
            this.updatePreview();
        });

        textInput.addEventListener('input', (e) => {
            const value = e.target.value;
            if (/^#[0-9A-Fa-f]{6}$/.test(value)) {
                this.template[templateKey] = value;
                colorInput.value = value;
                this.updatePreview();
            }
        });
    }

    bindSlider(name, callback) {
        const slider = document.getElementById(name + 'Slider');
        slider.addEventListener('input', (e) => {
            callback(parseFloat(e.target.value));
        });
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.btnPlay.textContent = this.isPlaying ? '⏸ Pause' : '▶ Play';
        this.btnPlay.classList.toggle('paused', this.isPlaying);

        if (this.isPlaying) {
            this.lastFrameTime = performance.now();
            this.animate();
        } else {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    animate() {
        if (!this.isPlaying) return;

        const now = performance.now();
        const delta = (now - this.lastFrameTime) / 1000;
        this.lastFrameTime = now;

        this.currentTime += delta;
        if (this.currentTime >= this.template.duration) {
            this.currentTime = 0;
        }

        this.updatePlayhead();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }

    updatePlayhead() {
        const percent = (this.currentTime / this.template.duration) * 100;
        this.playhead.style.left = percent + '%';

        const minutes = Math.floor(this.currentTime / 60);
        const seconds = (this.currentTime % 60).toFixed(1);
        this.currentTimeDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.padStart(4, '0')}`;
    }

    updatePreview() {
        const preview = this.captionPreview;
        const t = this.template;

        // Position
        preview.style.left = (t.pos_x * 100) + '%';
        preview.style.top = (t.pos_y * 100) + '%';

        // Text
        preview.textContent = t.caption_text;
        preview.style.color = t.text_color;
        preview.style.fontSize = t.font_size + 'px';

        // Style
        switch (t.style) {
            case 'box':
                preview.style.backgroundColor = 'rgba(0,0,0,0.7)';
                preview.style.borderColor = t.highlight_color;
                preview.style.borderWidth = '2px';
                preview.style.borderStyle = 'solid';
                preview.style.textShadow = 'none';
                break;
            case 'clean':
                preview.style.backgroundColor = 'transparent';
                preview.style.borderColor = 'transparent';
                preview.style.textShadow = 'none';
                break;
            case 'outline':
                preview.style.backgroundColor = 'transparent';
                preview.style.borderColor = 'transparent';
                preview.style.textShadow = `
                    -${t.stroke_width}px -${t.stroke_width}px 0 ${t.stroke_color},
                    ${t.stroke_width}px -${t.stroke_width}px 0 ${t.stroke_color},
                    -${t.stroke_width}px ${t.stroke_width}px 0 ${t.stroke_color},
                    ${t.stroke_width}px ${t.stroke_width}px 0 ${t.stroke_color}
                `;
                break;
            case 'shadow':
                preview.style.backgroundColor = 'transparent';
                preview.style.borderColor = 'transparent';
                preview.style.textShadow = '3px 3px 6px rgba(0,0,0,0.8)';
                break;
            case 'gradient':
                preview.style.backgroundColor = 'linear-gradient(135deg, rgba(0,0,0,0.8), rgba(57,229,95,0.3))';
                preview.style.borderColor = t.highlight_color;
                preview.style.borderWidth = '2px';
                preview.style.borderStyle = 'solid';
                preview.style.textShadow = 'none';
                break;
            case 'neon':
                preview.style.backgroundColor = 'transparent';
                preview.style.borderColor = 'transparent';
                preview.style.textShadow = `
                    0 0 5px ${t.highlight_color},
                    0 0 10px ${t.highlight_color},
                    0 0 20px ${t.highlight_color},
                    0 0 40px ${t.highlight_color}
                `;
                break;
        }
    }

    updateAspectRatio() {
        const ratios = {
            '9:16': { ratio: '9 / 16', width: 1080, height: 1920 },
            '1:1': { ratio: '1 / 1', width: 1080, height: 1080 },
            '16:9': { ratio: '16 / 9', width: 1920, height: 1080 },
            '4:5': { ratio: '4 / 5', width: 1080, height: 1350 },
            '4:3': { ratio: '4 / 3', width: 1440, height: 1080 }
        };

        const config = ratios[this.template.aspect_ratio] || ratios['1:1'];
        this.aspectContainer.style.aspectRatio = config.ratio;
        this.statSize.textContent = `${config.width}×${config.height}`;
    }

    updateTimeline() {
        // Update time ruler
        const duration = Math.ceil(this.template.duration);
        this.timeRuler.innerHTML = '';

        for (let i = 0; i <= duration; i++) {
            const marker = document.createElement('div');
            marker.className = 'time-marker';
            marker.innerHTML = `<span>${i}s</span>`;
            this.timeRuler.appendChild(marker);
        }

        // Update total time display
        this.totalTimeDisplay.textContent = this.template.duration.toFixed(1) + 's';
    }

    updateStats() {
        this.statDuration.textContent = this.template.duration.toFixed(1) + 's';
        this.statAspect.textContent = this.template.aspect_ratio;
    }

    copySettings() {
        const json = JSON.stringify(this.template, null, 2);
        navigator.clipboard.writeText(json).then(() => {
            alert('Settings copied to clipboard!');
        });
    }

    resetToDefault() {
        this.template = {
            caption_text: "Hello World",
            duration: 5.0,
            aspect_ratio: "1:1",
            style: "box",
            highlight_color: "#39E55F",
            text_color: "#FFFFFF",
            stroke_color: "#000000",
            stroke_width: 2,
            font_size: 24,
            pos_x: 0.5,
            pos_y: 0.8,
            fps: 60,
            quality: "Medium"
        };

        // Update all UI elements
        this.captionText.value = this.template.caption_text;
        this.highlightColor.value = this.template.highlight_color;
        this.highlightColorText.value = this.template.highlight_color;
        this.textColor.value = this.template.text_color;
        this.textColorText.value = this.template.text_color;
        this.strokeColor.value = this.template.stroke_color;
        this.strokeColorText.value = this.template.stroke_color;
        this.durationSlider.value = this.template.duration;
        this.durationValue.textContent = this.template.duration + 's';
        this.aspectSelect.value = this.template.aspect_ratio;
        this.posXSlider.value = this.template.pos_x;
        this.posXValue.textContent = this.template.pos_x.toFixed(2);
        this.posYSlider.value = this.template.pos_y;
        this.posYValue.textContent = this.template.pos_y.toFixed(2);
        this.fontSizeSlider.value = this.template.font_size;
        this.fontSizeValue.textContent = this.template.font_size + 'px';
        this.strokeWidthSlider.value = this.template.stroke_width;
        this.strokeWidthValue.textContent = this.template.stroke_width + 'px';

        // Reset style presets
        this.stylePresets.querySelectorAll('.style-preset').forEach(b => b.classList.remove('active'));
        this.stylePresets.querySelector('[data-style="box"]').classList.add('active');

        this.updatePreview();
        this.updateAspectRatio();
        this.updateTimeline();
        this.updateStats();
    }

    exportToComfyUI() {
        const workflow = {
            class_type: "CaptionLiveNode",
            inputs: {
                text: this.template.caption_text,
                duration: this.template.duration,
                aspect_ratio: this.template.aspect_ratio,
                style: this.template.style,
                highlight_color: this.template.highlight_color,
                text_color: this.template.text_color,
                stroke_color: this.template.stroke_color,
                stroke_width: this.template.stroke_width,
                font_size: this.template.font_size,
                pos_x: this.template.pos_x,
                pos_y: this.template.pos_y
            }
        };

        const json = JSON.stringify({ "1": workflow }, null, 2);

        // Create download
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'caption_workflow.json';
        a.click();
        URL.revokeObjectURL(url);
    }
}

// Initialize editor when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.editor = new CaptionEditor();
    console.log('🎬 Caption Live Editor initialized');
});
