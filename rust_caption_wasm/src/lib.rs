use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};
use serde::{Deserialize, Serialize};
use caption_core::{LayoutEngine, TextMeasurer, Rect, Word, Segment, parse_hex_rgba, interpolate_rect, LayoutSettings};

#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    Ok(())
}

struct CanvasMeasurer<'a> {
    ctx: &'a CanvasRenderingContext2d,
}

impl<'a> TextMeasurer for CanvasMeasurer<'a> {
    fn measure_text(&self, text: &str, font_size: f64) -> (f64, f64) {
        let font_str = format!("900 {}px Inter, Arial, sans-serif", font_size);
        self.ctx.set_font(&font_str);
        
        let width = self.ctx.measure_text(text).unwrap().width();
        let height = font_size * 1.3;
        
        (width, height)
    }
}

#[wasm_bindgen]
pub struct CaptionEngine {
    words: Vec<Word>,
    #[wasm_bindgen(skip)]
    pub gpu_renderer: Option<caption_core::gpu::GpuRenderer>,
    #[wasm_bindgen(skip)]
    pub device: Option<wgpu::Device>,
    #[wasm_bindgen(skip)]
    pub queue: Option<wgpu::Queue>,
    #[wasm_bindgen(skip)]
    pub surface: Option<wgpu::Surface<'static>>,
    #[wasm_bindgen(skip)]
    pub config: Option<wgpu::SurfaceConfiguration>,
}

#[wasm_bindgen]
impl CaptionEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(json_data: &str) -> CaptionEngine {
        let segments: Vec<Segment> = serde_json::from_str(json_data).unwrap_or_default();
        let mut all_words = Vec::new();
        for seg in segments {
            if let Some(words) = seg.words {
                all_words.extend(words);
            } else {
                all_words.push(Word {
                    text: seg.text,
                    start: seg.start,
                    end: seg.end,
                });
            }
        }
        CaptionEngine { 
            words: all_words,
            gpu_renderer: None,
            device: None,
            queue: None,
            surface: None,
            config: None,
        }
    }

    pub async fn init_gpu(&mut self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }).await.ok_or_else(|| JsValue::from_str("No suitable adapter found"))?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
            },
            None,
        ).await.map_err(|e| JsValue::from_str(&e.to_string()))?;

        let width = 800; // Initial dummy size, will reconfigure on draw
        let height = 600;
        let caps = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let renderer = caption_core::gpu::GpuRenderer::new(&device, &queue, config.format);

        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.config = Some(config);
        self.gpu_renderer = Some(renderer);

        Ok(())
    }

    pub fn draw_frame_gpu(&mut self, width: u32, height: u32, _time: f64, _style: &str, _font_size_px: f64, _pos_x: f64, _pos_y: f64, _highlight_color: &str, _text_color: &str, _font_family: &str) {
        if let (Some(device), Some(queue), Some(surface), Some(config), Some(renderer)) = 
            (&self.device, &self.queue, &self.surface, &mut self.config, &mut self.gpu_renderer) {
            
            if config.width != width || config.height != height {
                config.width = width;
                config.height = height;
                surface.configure(device, config);
            }

            let frame = surface.get_current_texture().expect("Failed to acquire next swap chain texture");
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            
            // Layout Logic (Reuse existing logic but need Measurer)
            // For now, we can't easily reuse CanvasMeasurer because we don't have ctx here.
            // We need a way to measure text without Canvas 2D context if we are in pure GPU mode.
            // But wait, we are in browser, we can create a temp canvas or just use the one passed?
            // Actually, for layout, we still need text metrics.
            // Glyphon has measuring capabilities!
            // But LayoutEngine expects a TextMeasurer trait.
            // I should implement TextMeasurer for Glyphon or just use a dummy one for now to test rendering?
            // No, layout is crucial.
            // I can create a temporary 2D context for measuring?
            // Or pass the 2D context just for measuring?
            
            // For this implementation, let's assume we pass the 2D context for measuring, 
            // but render to GPU.
            // But `draw_frame_gpu` doesn't take ctx.
            
            // Let's skip layout for a moment and just draw a test text to verify GPU works.
            // Or better, use a dummy layout.
            
            // ... (Rendering logic) ...
            
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                // Dummy commands for testing
                let commands = vec![
                    caption_core::DrawCommand::DrawText {
                        text: "GPU Rendering!".to_string(),
                        x: 50.0,
                        y: 50.0,
                        font_size: 40.0,
                        fill_color: "#FFFFFF".to_string(),
                        stroke_color: "none".to_string(),
                        stroke_width: 0.0,
                        scale: 1.0,
                        rotation: 0.0,
                    }
                ];

                renderer.render(device, queue, &mut pass, &commands, width, height);
            }

            queue.submit(Some(encoder.finish()));
            frame.present();
        }
    }

    fn path_round_rect(&self, ctx: &CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64, r: f64) {
        // Clamp radius to prevent artifacts when box is small
        let max_r = (w.min(h)) / 2.0;
        let r = r.min(max_r).max(0.0);

        ctx.begin_path();
        let _ = ctx.move_to(x + r, y);
        let _ = ctx.line_to(x + w - r, y);
        let _ = ctx.quadratic_curve_to(x + w, y, x + w, y + r);
        let _ = ctx.line_to(x + w, y + h - r);
        let _ = ctx.quadratic_curve_to(x + w, y + h, x + w - r, y + h);
        let _ = ctx.line_to(x + r, y + h);
        let _ = ctx.quadratic_curve_to(x, y + h, x, y + h - r);
        let _ = ctx.line_to(x, y + r);
        let _ = ctx.quadratic_curve_to(x, y, x + r, y);
        ctx.close_path();
    }

    pub fn draw_frame(&self, ctx: &CanvasRenderingContext2d, width: f64, height: f64, time: f64, style: &str, font_size_px: f64, pos_x: f64, pos_y: f64, highlight_color: &str, text_color: &str, font_family: &str) {
        // Fill with black background
        ctx.set_fill_style(&JsValue::from_str("#000000"));
        ctx.fill_rect(0.0, 0.0, width, height);

        // NO LETTERBOXING - stretch content to fill entire canvas
        // Use actual canvas dimensions for layout
        let side_margin = 20.0;
        let container_h = height * 0.80; // Use 80% of height to allow large text
        
        let container = Rect {
            x: side_margin, // WASM layout is relative to container, pos_x is applied later? 
                            // Actually, in Rust backend we applied pos_x to DrawCommand.
                            // Here, let's keep container static and let FrameGenerator handle pos_x/y offset?
                            // Wait, FrameGenerator takes pos_x/pos_y and adds it to rects.
                            // So container should be at (0,0) or centered?
                            // In Rust backend: container was (0,0,w,h).
                            // Here: container is centered vertically.
            y: (height - container_h) / 2.0, 
            w: width - (side_margin * 2.0), 
            h: container_h, 
        };

        let measurer = CanvasMeasurer { ctx };
        let engine = LayoutEngine::new(&self.words, &measurer);
        
        let settings = LayoutSettings {
            container,
            min_font_size: font_size_px,
            max_font_size: font_size_px,
            padding: 10.0,
        };
        
        let layout = engine.calculate_best_fit(settings);
        let font_size = layout.font_size;
        
        ctx.set_font(&format!("900 {}px {}, Inter, Arial, sans-serif", font_size, font_family));
        ctx.set_text_align("left");
        ctx.set_text_baseline("middle"); // Changed to middle to match SVG
        ctx.set_line_join("round");
        ctx.set_line_cap("round");

        // Animation state
        let frame_state = engine.calculate_frame(&layout, time);
        
        // Generate Commands
        let commands = caption_core::FrameGenerator::generate_frame(
            &layout, 
            &frame_state, 
            &self.words, 
            style, 
            highlight_color, 
            text_color, 
            pos_x, 
            pos_y
        );
        
        // Execute Commands
        for cmd in commands {
            match cmd {
                caption_core::DrawCommand::DrawRect { rect, color, radius } => {
                    ctx.set_fill_style(&JsValue::from_str(&color));
                    self.path_round_rect(ctx, rect.x, rect.y, rect.w, rect.h, radius);
                    ctx.fill();
                },
                caption_core::DrawCommand::DrawText { text, x, y, font_size: _, fill_color, stroke_color, stroke_width, scale, rotation: _ } => {
                    // Note: font_size is already set on ctx, assuming it doesn't change per word (it doesn't in current logic)
                    // But if FrameGenerator allows different sizes, we should set it.
                    // For now, layout.font_size is constant.
                    
                    ctx.set_fill_style(&JsValue::from_str(&fill_color));
                    
                    if scale != 1.0 {
                        let _ = ctx.save();
                        // We need center to scale around. 
                        // Text is drawn at x,y (left, middle).
                        // We need to measure text to find center? 
                        // Or just translate to x,y and scale? 
                        // If we translate to x,y, we scale from the left edge.
                        // To scale from center, we need width.
                        let width = ctx.measure_text(&text).unwrap().width();
                        let cx = x + width / 2.0;
                        let cy = y; // Baseline is middle
                        
                        let _ = ctx.translate(cx, cy);
                        let _ = ctx.scale(scale, scale);
                        let _ = ctx.translate(-cx, -cy);
                        
                        if stroke_width > 0.0 {
                            ctx.set_line_width(stroke_width);
                            ctx.set_stroke_style(&JsValue::from_str(&stroke_color));
                            let _ = ctx.stroke_text(&text, x, y);
                        }
                        let _ = ctx.fill_text(&text, x, y);
                        
                        let _ = ctx.restore();
                    } else {
                        if stroke_width > 0.0 {
                            ctx.set_line_width(stroke_width);
                            ctx.set_stroke_style(&JsValue::from_str(&stroke_color));
                            let _ = ctx.stroke_text(&text, x, y);
                        }
                        let _ = ctx.fill_text(&text, x, y);
                    }
                }
            }
        }
    }
}
