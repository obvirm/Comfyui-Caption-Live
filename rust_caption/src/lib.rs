use pyo3::prelude::*;
use tiny_skia::{Pixmap, Transform, PixmapPaint, IntSize, Paint, Color};
use usvg::{self, Options, TreeParsing, TreeTextToPath};
use caption_core::{Word, LayoutEngine, LayoutSettings, TextMeasurer, Rect};
use rusttype::{Font, Scale};

// Implement TextMeasurer using rusttype
struct RustTypeMeasurer<'a> {
    font: &'a Font<'a>,
}

impl<'a> TextMeasurer for RustTypeMeasurer<'a> {
    fn measure_text(&self, text: &str, font_size: f64) -> (f64, f64) {
        let scale = Scale::uniform(font_size as f32);
        let v_metrics = self.font.v_metrics(scale);
        
        let mut width = 0.0;
        for glyph in self.font.layout(text, scale, rusttype::point(0.0, 0.0)) {
            if let Some(bb) = glyph.pixel_bounding_box() {
                width = bb.max.x as f64;
            }
        }
        
        let height = (v_metrics.ascent - v_metrics.descent) as f64;
        (width, height)
    }
}

// Helper to draw a rounded rect using SVG
fn draw_rounded_rect_svg(rect: Rect, color: &str, radius: f64) -> String {
    // Clamp radius to prevent artifacts when box is small (Match WASM logic)
    let max_r = (rect.w.min(rect.h)) / 2.0;
    let r = radius.min(max_r).max(0.0);

    format!(
        r#"<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="{}" />"#,
        rect.x, rect.y, rect.w, rect.h, r, color
    )
}

// Helper to draw text using SVG
fn draw_text_svg(text: &str, x: f64, y: f64, font_family: &str, font_size: f64, color: &str, stroke_color: &str, stroke_width: f64) -> String {
    let escaped_text = text.replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;")
                           .replace("\"", "&quot;")
                           .replace("'", "&apos;");
                           
    format!(
        r#"
        <text x="{}" y="{}" font-family="{}" font-size="{}" fill="{}" stroke="{}" stroke-width="{}" stroke-linejoin="round" paint-order="stroke" font-weight="900" dominant-baseline="middle">
            {}
        </text>
        "#,
        x, y, font_family, font_size, color, stroke_color, stroke_width, escaped_text
    )
}

#[pyfunction]
fn render_tiktok_batch(
    _py: Python,
    py_images: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    fps: f64,
    subtitle_json: String,
    font_path: String,
    style: String, 
    highlight_color: String,
    text_color: String,
    pos_x: f64, 
    pos_y: f64,
    font_size: f64
) -> PyResult<Vec<Vec<u8>>> {
    
    // 1. Load Font for Measuring (RustType)
    let font_data = std::fs::read(&font_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read font file: {}", e))
    })?;
    let font = Font::try_from_vec(font_data.clone()).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to parse font data")
    })?;
    
    let measurer = RustTypeMeasurer { font: &font };

    // 2. Setup Font Database for Rendering (USVG)
    let mut fontdb = usvg::fontdb::Database::new();
    fontdb.load_system_fonts();
    fontdb.load_font_data(font_data); 
    let font_family = "Montserrat"; 
    
    // 3. Parse Subtitles -> Words
    let segments: Vec<serde_json::Value> = serde_json::from_str(&subtitle_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON Error: {}", e))
    })?;
    
    let mut words = Vec::new();
    for seg in segments {
        let text = seg["text"].as_str().unwrap_or("").to_string();
        let start = seg["start"].as_f64().unwrap_or(0.0);
        let end = seg["end"].as_f64().unwrap_or(0.0);
        words.push(Word { text, start, end });
    }
    
    // 4. Initialize Layout Engine
    let engine = LayoutEngine::new(&words, &measurer);
    
    // Define Layout Settings (Match WASM logic)
    // NO LETTERBOXING - stretch content to fill entire canvas
    // Use actual canvas dimensions for layout
    let side_margin = 20.0;
    let container_h = (height as f64) * 0.80; // Use 80% of height to allow large text
    
    let container = Rect {
        x: side_margin, 
        y: ((height as f64) - container_h) / 2.0,
        w: (width as f64) - (side_margin * 2.0),
        h: container_h,
    };
    
    let settings = LayoutSettings {
        container,
        min_font_size: font_size, // Fixed font size
        max_font_size: font_size,
        padding: 10.0, // Match WASM padding
    };
    
    let layout = engine.calculate_best_fit(settings);
    
    let mut output_images = Vec::with_capacity(py_images.len());

    // Loop Processing
    for (i, raw_bytes) in py_images.into_iter().enumerate() {
        let current_time = i as f64 / fps;
        
        let mut final_frame_data = raw_bytes;
        
        {
            let size = IntSize::from_wh(width, height).ok_or_else(|| {
                 PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid dimensions")
            })?;

            let mut pixmap = Pixmap::from_vec(final_frame_data, size).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to create Pixmap from bytes")
            })?;

            // Calculate Frame State (Animation)
            let frame_state = engine.calculate_frame(&layout, current_time);
            
            // Generate Draw Commands from Core
            let commands = caption_core::FrameGenerator::generate_frame(
                &layout, 
                &frame_state, 
                &words, 
                &style, 
                &highlight_color, 
                &text_color, 
                pos_x, 
                pos_y
            );
            
            // Build SVG from Commands
            let mut svg_content = String::new();
            
            for cmd in commands {
                match cmd {
                    caption_core::DrawCommand::DrawRect { rect, color, radius } => {
                        svg_content.push_str(&draw_rounded_rect_svg(rect, &color, radius));
                    },
                    caption_core::DrawCommand::DrawText { text, x, y, font_size, fill_color, stroke_color, stroke_width, scale, rotation: _ } => {
                        let mut final_font_size = font_size;
                        if scale != 1.0 {
                            final_font_size = font_size * scale;
                        }

                        // Emulate Canvas "Stroke then Fill" by drawing two text elements
                        // 1. Stroke Layer (Background)
                        if stroke_width > 0.0 {
                            svg_content.push_str(&format!(
                                r#"<text x="{}" y="{}" font-family="{}" font-size="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linejoin="round" font-weight="900" dominant-baseline="middle">{}</text>"#,
                                x, y, font_family, final_font_size, stroke_color, stroke_width, text
                            ));
                        }

                        // 2. Fill Layer (Foreground)
                        svg_content.push_str(&format!(
                            r#"<text x="{}" y="{}" font-family="{}" font-size="{}" fill="{}" stroke="none" font-weight="900" dominant-baseline="middle">{}</text>"#,
                            x, y, font_family, final_font_size, fill_color, text
                        ));
                    }
                }
            }
            
            let svg_data = format!(
                r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">{}</svg>"#,
                width, height, svg_content
            );
            
            // Render SVG
            let mut opt = Options::default();
            opt.font_family = font_family.to_string();
            
            if let Ok(mut tree) = usvg::Tree::from_str(&svg_data, &opt) {
                tree.convert_text(&fontdb);
                let mut text_pixmap = Pixmap::new(width, height).unwrap();
                let rtree = resvg::Tree::from_usvg(&tree);
                rtree.render(Transform::default(), &mut text_pixmap.as_mut());
                
                pixmap.draw_pixmap(
                    0, 0, 
                    text_pixmap.as_ref(), 
                    &PixmapPaint::default(), 
                    Transform::default(), 
                    None
                );
            }
            
            final_frame_data = pixmap.take();
        }
        
        output_images.push(final_frame_data);
    }
    output_images.shrink_to_fit();
    Ok(output_images)
}

#[cfg(feature = "gpu")]
#[pyfunction]
fn render_gpu(
    _py: Python,
    py_images: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    fps: f64,
    subtitle_json: String,
    font_path: String,
    style: String,
    highlight_color: String,
    text_color: String,
    pos_x: f64,
    pos_y: f64,
    font_size: f64
) -> PyResult<Vec<Vec<u8>>> {
    // Parse Subtitles (Reuse logic or extract to helper)
    let segments: Vec<serde_json::Value> = serde_json::from_str(&subtitle_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON Error: {}", e))
    })?;
    
    let mut words = Vec::new();
    for seg in segments {
        let text = seg["text"].as_str().unwrap_or("").to_string();
        let start = seg["start"].as_f64().unwrap_or(0.0);
        let end = seg["end"].as_f64().unwrap_or(0.0);
        words.push(Word { text, start, end });
    }

    // Layout Engine (Need Measurer)
    // For GPU, we might use glyphon for measuring too?
    // Or just use RustTypeMeasurer as before for layout calculation.
    // Ideally, layout should match rendering. Glyphon uses cosmic-text which has its own shaping.
    // If we use RustType for layout and Glyphon for rendering, they might mismatch.
    // But for now, let's reuse RustTypeMeasurer for layout to keep it simple.
    
    let font_data = std::fs::read(&font_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read font file: {}", e))
    })?;
    let font = Font::try_from_vec(font_data.clone()).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to parse font data")
    })?;
    let measurer = RustTypeMeasurer { font: &font };
    let engine = LayoutEngine::new(&words, &measurer);

    // Layout Settings
    let side_margin = 20.0;
    let container_h = (height as f64) * 0.80;
    let container = Rect {
        x: side_margin, 
        y: ((height as f64) - container_h) / 2.0,
        w: (width as f64) - (side_margin * 2.0),
        h: container_h,
    };
    let settings = LayoutSettings {
        container,
        min_font_size: font_size,
        max_font_size: font_size,
        padding: 10.0,
    };
    let layout = engine.calculate_best_fit(settings);

    // GPU Context
    let mut output_images = Vec::with_capacity(py_images.len());

    pollster::block_on(async {
        let mut ctx = caption_core::gpu::HeadlessContext::new(width, height).await;

        for (i, raw_bytes) in py_images.into_iter().enumerate() {
            let current_time = i as f64 / fps;
            let frame_state = engine.calculate_frame(&layout, current_time);
            
            let commands = caption_core::FrameGenerator::generate_frame(
                &layout, 
                &frame_state, 
                &words, 
                &style, 
                &highlight_color, 
                &text_color, 
                pos_x, 
                pos_y
            );

            let result_bytes = ctx.render_frame(&raw_bytes, &commands).await;
            output_images.push(result_bytes);
        }
    });

    Ok(output_images)
}

#[pymodule]
fn rust_caption(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_tiktok_batch, m)?)?;
    #[cfg(feature = "gpu")]
    m.add_function(wrap_pyfunction!(render_gpu, m)?)?;
    Ok(())
}
