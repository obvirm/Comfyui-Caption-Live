use pyo3::prelude::*;
use image::{Rgba, RgbaImage, DynamicImage, GenericImageView};
use imageproc::drawing::{draw_text_mut, draw_filled_rect_mut, draw_filled_circle_mut};
use imageproc::rect::Rect as ImageRect;
use rusttype::{Font, Scale, point, PositionedGlyph};
use std::fs;
use std::path::Path;
use caption_core::{LayoutEngine, TextMeasurer, Rect, Word, Segment, parse_hex_rgba, interpolate_rect, LayoutSettings};

// Implement TextMeasurer for rusttype::Font
struct RusttypeMeasurer<'a> {
    font: &'a Font<'a>,
}

impl<'a> TextMeasurer for RusttypeMeasurer<'a> {
    fn measure_text(&self, text: &str, font_size: f64) -> (f64, f64) {
        let scale = Scale::uniform(font_size as f32);
        let v_metrics = self.font.v_metrics(scale);
        let height = (v_metrics.ascent - v_metrics.descent + v_metrics.line_gap) as f64;
        
        let width = self.font
            .layout(text, scale, point(0.0, 0.0))
            .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
            .last()
            .unwrap_or(0.0) as f64;
            
        (width, height)
    }
}

fn draw_rounded_rect_mut(image: &mut RgbaImage, rect: Rect, radius: f64, color: Rgba<u8>) {
    let x = rect.x as i32;
    let y = rect.y as i32;
    let w = rect.w as i32;
    let h = rect.h as i32;
    let r = radius as i32;

    // Center rect
    draw_filled_rect_mut(image, ImageRect::at(x + r, y).of_size((w - 2 * r) as u32, h as u32), color);
    // Left rect
    draw_filled_rect_mut(image, ImageRect::at(x, y + r).of_size(r as u32, (h - 2 * r) as u32), color);
    // Right rect
    draw_filled_rect_mut(image, ImageRect::at(x + w - r, y + r).of_size(r as u32, (h - 2 * r) as u32), color);
    
    // Corners
    draw_filled_circle_mut(image, (x + r, y + r), r, color);
    draw_filled_circle_mut(image, (x + w - r, y + r), r, color);
    draw_filled_circle_mut(image, (x + r, y + h - r), r, color);
    draw_filled_circle_mut(image, (x + w - r, y + h - r), r, color);
}

#[pyfunction]
fn render_mask_v2(
    json_data: String,
    width: u32,
    height: u32,
    time: f64,
    _fps: f64,
    font_path: String,
    style: String,
    highlight_color: String,
    text_color: String,
    pos_x: f64,
    pos_y: f64,
    font_size_px: f64, 
) -> PyResult<Vec<u8>> {
    
    // 1. Load Font
    let font_data = fs::read(&font_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load font: {}", e)))?;
    let font = Font::try_from_vec(font_data).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Error constructing font"))?;
    println!("[RUST] Font loaded. Font_path: {}", font_path);

    // 2. Parse Data
    println!("[RUST] Raw json_data received: {}", json_data);
    let segments: Vec<Segment> = serde_json::from_str(&json_data).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON Error: {}", e)))?;
    println!("[RUST] Segments parsed. Count: {}", segments.len());
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
    println!("[RUST] All words collected. Count: {}", all_words.len());

    // Check for empty words before proceeding
    if all_words.is_empty() {
        println!("[RUST] WARNING: No words found after parsing segments. Returning empty image.");
        return Ok(RgbaImage::new(width, height).into_vec());
    }

    // 3. Setup Engine & Layout
    let measurer = RusttypeMeasurer { font: &font };
    let engine = LayoutEngine::new(&all_words, &measurer);
    println!("[RUST] LayoutEngine created with {} words.", all_words.len());
    
    // Layout Logic (Matching WASM v2)
    // Use actual pixel dimensions without arbitrary scaling
    let w_f64 = width as f64;
    let h_f64 = height as f64;
    
    let side_margin = 20.0;
    let container_h = h_f64 * 0.80; // 80% height
    
    let container = Rect {
        x: side_margin + pos_x,
        y: ((h_f64 - container_h) / 2.0) + pos_y,
        w: w_f64 - (side_margin * 2.0),
        h: container_h,
    };
    println!("[RUST] Container Rect: x={}, y={}, w={}, h={}", container.x, container.y, container.w, container.h);
    
    // Fixed font size requested by user
    let settings = LayoutSettings {
        container,
        min_font_size: font_size_px,
        max_font_size: font_size_px,
        padding: 10.0,
    };
    
    let layout = engine.calculate_best_fit(settings);
    let font_size = layout.font_size;
    println!("[RUST] Layout calculated. Layout words count: {}, Font Size: {}", layout.words.len(), layout.font_size);
    if layout.words.is_empty() {
        println!("[RUST] WARNING: Layout resulted in no words. Returning empty image.");
        return Ok(RgbaImage::new(width, height).into_vec());
    }
    
    // 4. Calculate Frame State (Animation & Active Word)
    let frame_state = engine.calculate_frame(&layout, time);
    let active_idx = frame_state.active_word_index;
    println!("[RUST] Frame state calculated. Active idx: {:?}", active_idx);

    // 5. Render to Image
    let mut image = RgbaImage::new(width, height);
    println!("[RUST] Image buffer created: {}x{} pixels", width, height);
    
    let col_text = Rgba(parse_hex_rgba(&text_color));
    let col_highlight = Rgba(parse_hex_rgba(&highlight_color));
    let col_white = Rgba([255, 255, 255, 255]);
    let col_black = Rgba([0, 0, 0, 255]);

    println!("[RUST] Colors - Text: {:?}, Highlight: {:?}", col_text, col_highlight);

    // A. Draw Passive Text
    // Use user-defined text_color for passive text instead of hardcoded black/white
    let passive_color = col_text; 
    
    for (i, item) in layout.words.iter().enumerate() {
        if let Some(active) = active_idx {
            // ALWAYS skip active word from passive drawing
            if i == active {
                continue;
            }
        }
        let word = &all_words[item.word_index];
        
        let x = item.rect.x as i32;
        let y = item.rect.y as i32;
        let scaled_scale = Scale::uniform(font_size as f32);

        println!("[RUST] Drawing passive text '{}' at ({},{})", word.text, x, y);
        draw_text_mut(&mut image, passive_color, x, y, scaled_scale, &font, &word.text);
    }

    // B. Draw Active Text & Box
    if let Some(idx) = active_idx {
        println!("[RUST] Active word index: {}", idx);
        if style == "box" {
            let target_item = &layout.words[idx];
            let word = &all_words[target_item.word_index];
            
            // Use calculated box rect from core engine
            if let Some(box_rect) = frame_state.box_rect {
                let box_h = box_rect.h;
                let box_radius = box_h * 0.2; // Dynamic radius
                
                println!("[RUST] Drawing box rect: x={}, y={}, w={}, h={}", box_rect.x, box_rect.y, box_rect.w, box_rect.h);
                // Use highlight_color for the box background
                draw_rounded_rect_mut(&mut image, box_rect, box_radius, col_highlight);
            }
            
            // Draw Active Word
            let x = target_item.rect.x as i32;
            let y = target_item.rect.y as i32;
            let scaled_scale = Scale::uniform(font_size as f32);
            
            println!("[RUST] Drawing active text '{}' for box style at ({},{})", word.text, x, y);
            // Use text_color (col_text) for active text inside box
            draw_text_mut(&mut image, col_text, x, y, scaled_scale, &font, &word.text);

        } else {
            // Colored / Scaling
            let item = &layout.words[idx];
            let word = &all_words[item.word_index];
            
            let x = item.rect.x as i32;
            let y = item.rect.y as i32;
            
            let effect_scale = if style == "scaling" { 1.25 } else { 1.1 };
            let final_scale = Scale::uniform((font_size * effect_scale) as f32);
            
            let (w_orig, h_orig) = measurer.measure_text(&word.text, font_size);
            let (w_new, h_new) = measurer.measure_text(&word.text, font_size * effect_scale);
            let offset_x = (w_new - w_orig) / 2.0;
            let offset_y = (h_new - h_orig) / 2.0;
            
            let draw_x = x - offset_x as i32;
            let draw_y = y - offset_y as i32;

            println!("[RUST] Drawing active text '{}' for colored/scaling style at ({},{})", word.text, draw_x, draw_y);
            // Stroke
            let stroke_dist = (font_size * 0.05).ceil() as i32;
            for oy in -stroke_dist..=stroke_dist {
                for ox in -stroke_dist..=stroke_dist {
                    if ox == 0 && oy == 0 { continue; }
                    draw_text_mut(&mut image, col_black, draw_x + ox, draw_y + oy, final_scale, &font, &word.text);
                }
            }

            // Fill
            draw_text_mut(&mut image, col_highlight, draw_x, draw_y, final_scale, &font, &word.text);
        }
    }

    Ok(image.into_vec())
}

#[pymodule]
fn rust_caption(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_mask_v2, m)?)?;
    Ok(())
}
