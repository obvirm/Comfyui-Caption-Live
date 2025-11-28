use wasm_bindgen::prelude::*;
use web_sys::CanvasRenderingContext2d;
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
        CaptionEngine { words: all_words }
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

    pub fn draw_frame(&self, ctx: &CanvasRenderingContext2d, width: f64, height: f64, time: f64, style: &str, font_size_px: f64, pos_x: f64, pos_y: f64) {
        // Fill with black background
        ctx.set_fill_style(&JsValue::from_str("#000000"));
        ctx.fill_rect(0.0, 0.0, width, height);

        // NO LETTERBOXING - stretch content to fill entire canvas
        // Use actual canvas dimensions for layout
        let side_margin = 20.0;
        let container_h = height * 0.80; // Use 80% of height to allow large text
        
        let container = Rect {
            x: side_margin + pos_x,
            y: ((height - container_h) / 2.0) + pos_y, 
            w: width - (side_margin * 2.0), 
            h: container_h, 
        };

        let measurer = CanvasMeasurer { ctx };
        let engine = LayoutEngine::new(&self.words, &measurer);
        
        // Fixed font size based on user input (scaled by DPR in JS)
        let settings = LayoutSettings {
            container,
            min_font_size: font_size_px,
            max_font_size: font_size_px,
            padding: 10.0,
        };
        
        
        let layout = engine.calculate_best_fit(settings);
        let font_size = layout.font_size;
        
        ctx.set_font(&format!("900 {}px Inter, Arial, sans-serif", font_size));
        ctx.set_text_align("left");
        ctx.set_text_baseline("top"); 
        ctx.set_line_join("round");
        ctx.set_line_cap("round");

        // Animation state
        let frame_state = engine.calculate_frame(&layout, time);
        let active_idx = frame_state.active_word_index;

        // Passive text
        ctx.set_fill_style(&JsValue::from_str("white"));
        if style == "colored" || style == "scaling" {
             ctx.set_fill_style(&JsValue::from_str("white")); 
        }

        for (i, item) in layout.words.iter().enumerate() {
            if let Some(active) = active_idx {
                if i == active && (style == "scaling" || style == "colored") {
                    continue;
                }
            }
            let word = &self.words[item.word_index];
            let _ = ctx.fill_text(&word.text, item.rect.x, item.rect.y);
        }

        // Active text/box
        if let Some(idx) = active_idx {
            if style == "box" {
                let target_item = &layout.words[idx];
                let word = &self.words[target_item.word_index];
                
                if let Some(box_rect) = frame_state.box_rect {
                    let box_h = box_rect.h;
                    // Dynamic radius relative to box size (20%)
                    let box_radius = box_h * 0.2; 

                    ctx.set_fill_style(&JsValue::from_str("#007AFF"));
                    self.path_round_rect(ctx, box_rect.x, box_rect.y, box_rect.w, box_rect.h, box_radius);
                    ctx.fill();

                    let _ = ctx.save();
                    self.path_round_rect(ctx, box_rect.x, box_rect.y, box_rect.w, box_rect.h, box_radius);
                    ctx.clip();

                    ctx.set_fill_style(&JsValue::from_str("white"));
                    for item in &layout.words {
                        let word = &self.words[item.word_index];
                        let item_box = Rect { 
                            x: item.rect.x, y: item.rect.y - font_size, 
                            w: item.rect.w, h: font_size * 2.0 
                        };
                        if item_box.x < box_rect.x + box_rect.w && item_box.x + item_box.w > box_rect.x &&
                           item_box.y < box_rect.y + box_rect.h && item_box.y + item_box.h > box_rect.y {
                               let _ = ctx.fill_text(&word.text, item.rect.x, item.rect.y);
                           }
                    }
                    let _ = ctx.restore();
                }

            } else {
                let item = &layout.words[idx];
                let word = &self.words[item.word_index];
                
                let _ = ctx.save();
                let cx = item.rect.x + (item.rect.w / 2.0);
                let cy = item.rect.y;
                
                let _ = ctx.translate(cx, cy);
                if style == "scaling" {
                    let _ = ctx.scale(1.25, 1.25);
                } else {
                    let _ = ctx.scale(1.1, 1.1);
                }
                let _ = ctx.translate(-cx, -cy);

                let stroke_width = font_size * 0.08;
                ctx.set_line_width(stroke_width);
                ctx.set_stroke_style(&JsValue::from_str("black"));
                let _ = ctx.stroke_text(&word.text, item.rect.x, item.rect.y);

                ctx.set_fill_style(&JsValue::from_str("#39E55F"));
                let _ = ctx.fill_text(&word.text, item.rect.x, item.rect.y);
                
                let _ = ctx.restore();
            }
        }
    }
}
