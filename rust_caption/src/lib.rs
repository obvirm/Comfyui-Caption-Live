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
fn render_mask(
    json_data: String,
    width: u32,
    height: u32,
    time: f64,
    fps: f64,
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

    // 2. Parse Data
    let segments: Vec<Segment> = serde_json::from_str(&json_data).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON Error: {}", e)))?;
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

    // 3. Setup Engine & Layout
    let measurer = RusttypeMeasurer { font: &font };
    let engine = LayoutEngine::new(&all_words, &measurer);
    
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
    
    // Fixed font size requested by user
    let settings = LayoutSettings {
        container,
        min_font_size: font_size_px,
        max_font_size: font_size_px,
        padding: 10.0,
    };
    
    let layout = engine.calculate_best_fit(settings);
    let font_size = layout.font_size; // Should be equal to font_size_px
    
    // 4. Determine Active Word
    let mut active_idx = None;
    for (i, item) in layout.words.iter().enumerate() {
        let word = &all_words[item.word_index];
        if time >= word.start && time < word.end {
            active_idx = Some(i);
            break;
        }
    }
    if active_idx.is_none() && !all_words.is_empty() && time >= all_words.last().unwrap().end {
         active_idx = Some(layout.words.len() - 1);
    }

    // 5. Render to Image
    let mut image = RgbaImage::new(width, height);
    
    let col_text = Rgba(parse_hex_rgba(&text_color));
    let col_highlight = Rgba(parse_hex_rgba(&highlight_color));
    let col_box = Rgba([0, 122, 255, 255]); // Blue
    let col_white = Rgba([255, 255, 255, 255]);
    let col_black = Rgba([0, 0, 0, 255]);

    // A. Draw Passive Text
    let passive_color = if style == "colored" || style == "scaling" { col_white } else { col_black };
    
    for (i, item) in layout.words.iter().enumerate() {
        if let Some(active) = active_idx {
            if i == active && (style == "scaling" || style == "colored") {
                continue;
            }
        }
        let word = &all_words[item.word_index];
        
        let x = item.rect.x as i32;
        let y = item.rect.y as i32;
        let scaled_scale = Scale::uniform(font_size as f32);

        draw_text_mut(&mut image, passive_color, x, y, scaled_scale, &font, &word.text);
    }

    // B. Draw Active Text & Box
    if let Some(idx) = active_idx {
        if style == "box" {
            let target_item = &layout.words[idx];
            let word = &all_words[target_item.word_index];
            
            let box_padding = font_size * 0.2;
            let box_h = font_size * 1.1;
            let box_radius = box_h * 0.2; // Dynamic radius

            let target_rect = Rect {
                x: target_item.rect.x - box_padding,
                y: target_item.rect.y - (box_h / 2.0),
                w: target_item.rect.w + (box_padding * 2.0),
                h: box_h,
            };

            let mut box_rect = target_rect;

            if idx > 0 {
                let prev_item = &layout.words[idx - 1];
                let transition_duration = 0.25;
                let time_into_word = time - word.start;

                if time_into_word < transition_duration {
                    let prev_rect = Rect {
                        x: prev_item.rect.x - box_padding,
                        y: prev_item.rect.y - (box_h / 2.0),
                        w: prev_item.rect.w + (box_padding * 2.0),
                        h: box_h,
                    };

                    if (prev_rect.y - target_rect.y).abs() > 1.0 {
                        box_rect = target_rect;
                    } else {
                        let t = (time_into_word / transition_duration).clamp(0.0, 1.0);
                        box_rect = interpolate_rect(prev_rect, target_rect, t);
                    }
                }
            }

            // Draw Box
            draw_rounded_rect_mut(&mut image, box_rect, box_radius, col_box);
            
            // Draw Active Word
            let x = target_item.rect.x as i32;
            let y = target_item.rect.y as i32;
            let scaled_scale = Scale::uniform(font_size as f32);
            
            draw_text_mut(&mut image, col_white, x, y, scaled_scale, &font, &word.text);

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
    m.add_function(wrap_pyfunction!(render_mask, m)?)?;
    Ok(())
}
