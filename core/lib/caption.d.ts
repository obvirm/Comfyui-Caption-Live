/**
 * Caption Engine - JavaScript/TypeScript bindings for WASM
 */

export interface Position {
    x: number;
    y: number;
}

export interface TextStyle {
    font_family: string;
    font_size: number;
    font_weight: number;
    color: string;
    stroke_color?: string;
    stroke_width: number;
}

export interface Segment {
    text: string;
    start: number;
    end: number;
}

export interface Animation {
    type: 'none' | 'box_highlight' | 'typewriter' | 'bounce' | 'colored';
    segments?: Segment[];
    box_color?: string;
    box_radius?: number;
    box_padding?: number;
    chars_per_second?: number;
    scale?: number;
    duration?: number;
    active_color?: string;
}

export interface TextLayer {
    type: 'text';
    content: string;
    style: TextStyle;
    position: Position;
    animation?: Animation;
}

export interface ImageLayer {
    type: 'image';
    src: string;
    position: Position;
    width?: number;
    height?: number;
}

export type Layer = TextLayer | ImageLayer;

export interface Canvas {
    width: number;
    height: number;
}

export interface Template {
    canvas: Canvas;
    duration: number;
    fps: number;
    layers: Layer[];
}

export interface FrameData {
    pixels: Uint8Array;
    width: number;
    height: number;
    timestamp: number;
}

export type Quality = 'draft' | 'preview' | 'final';

declare class CaptionEngine {
    constructor();

    /**
     * Initialize the engine with dimensions
     */
    init(width: number, height: number): Promise<void>;

    /**
     * Render a frame from template JSON
     */
    renderFrame(templateJson: string, time: number): FrameData;

    /**
     * Render frame to ImageData for canvas display
     */
    renderToImageData(templateJson: string, time: number): ImageData;

    /**
     * Compute hash for frame validation
     */
    computeHash(templateJson: string, time: number): bigint;

    /**
     * Get current backend name
     */
    getBackend(): string;

    /**
     * Check if WebGPU is available
     */
    isWebGPUAvailable(): boolean;
}

/**
 * Load and initialize the Caption Engine WASM module
 */
export function loadCaptionEngine(): Promise<CaptionEngine>;

/**
 * Create a template from simple parameters
 */
export function createTemplate(options: {
    text: string;
    animation?: Animation['type'];
    style?: Partial<TextStyle>;
    position?: Partial<Position>;
    canvas?: Partial<Canvas>;
    duration?: number;
    fps?: number;
}): Template;

/**
 * Validate a template JSON string
 */
export function validateTemplate(json: string): { valid: boolean; errors: string[] };
