use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Segment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub words: Option<Vec<Word>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Clone, Debug)]
pub struct LayoutSettings {
    pub container: Rect,
    pub min_font_size: f64,
    pub max_font_size: f64,
    pub padding: f64,
}

#[derive(Clone, Debug)]
pub struct LayoutWord {
    pub word_index: usize,
    pub rect: Rect,
}

#[derive(Debug)]
pub struct LayoutResult {
    pub words: Vec<LayoutWord>,
    pub font_size: f64,
}

pub trait TextMeasurer {
    fn measure_text(&self, text: &str, font_size: f64) -> (f64, f64);
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

pub fn ease_out_cubic(t: f64) -> f64 {
    1.0 - (1.0 - t).powi(3)
}

pub fn interpolate_rect(from: Rect, to: Rect, t: f64) -> Rect {
    let eased_t = ease_out_cubic(t);
    Rect {
        x: lerp(from.x, to.x, eased_t),
        y: lerp(from.y, to.y, eased_t),
        w: lerp(from.w, to.w, eased_t),
        h: lerp(from.h, to.h, eased_t),
    }
}

pub fn parse_hex_rgba(hex: &str) -> [u8; 4] {
    let hex = hex.trim_start_matches('#');
    if hex.len() == 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(255);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255);
        [r, g, b, 255]
    } else {
        [255, 255, 255, 255]
    }
}

pub struct LayoutEngine<'a, M: TextMeasurer> {
    words: &'a [Word],
    measurer: &'a M,
}

impl<'a, M: TextMeasurer> LayoutEngine<'a, M> {
    pub fn new(words: &'a [Word], measurer: &'a M) -> Self {
        Self { words, measurer }
    }

    fn try_layout(&self, font_size: f64, settings: &LayoutSettings) -> Option<LayoutResult> {
        let container = settings.container;
        let padding = settings.padding;
        
        let space_width = self.measurer.measure_text(" ", font_size).0;
        let line_height = font_size * 1.3;
        
        let max_w = container.w - (padding * 2.0);
        let max_h = container.h - (padding * 2.0);

        let mut lines: Vec<Vec<LayoutWord>> = Vec::new();
        let mut current_line: Vec<LayoutWord> = Vec::new();
        let mut current_line_width = 0.0;
        let mut current_y = 0.0;

        for (i, word) in self.words.iter().enumerate() {
            let (word_w, _) = self.measurer.measure_text(&word.text, font_size);
            
            if current_line_width + word_w > max_w {
                if !current_line.is_empty() {
                    lines.push(current_line);
                    current_line = Vec::new();
                    current_line_width = 0.0;
                    current_y += line_height;
                    
                    if current_y + line_height > max_h {
                        return None;
                    }
                }
            }
            
            current_line.push(LayoutWord {
                word_index: i,
                rect: Rect { x: current_line_width, y: current_y, w: word_w, h: line_height },
            });
            current_line_width += word_w + space_width;
        }
        
        if !current_line.is_empty() {
            lines.push(current_line);
            current_y += line_height;
        }

        if current_y > max_h {
            return None;
        }

        let total_text_h = current_y;
        let start_y_offset = container.y + padding + (max_h - total_text_h) / 2.0;

        let mut final_words = Vec::new();
        for line in lines {
            if line.is_empty() { continue; }
            let last_word = line.last().unwrap();
            let line_content_width = last_word.rect.x + last_word.rect.w;
            
            let start_x_offset = container.x + padding + (max_w - line_content_width) / 2.0;
            
            for mut item in line {
                item.rect.x += start_x_offset;
                item.rect.y += start_y_offset;
                final_words.push(item);
            }
        }

        Some(LayoutResult {
            words: final_words,
            font_size,
        })
    }

    pub fn calculate_best_fit(&self, settings: LayoutSettings) -> LayoutResult {
        let min_font = settings.min_font_size;
        let max_font = settings.max_font_size;
        
        let mut low = min_font;
        let mut high = max_font;
        let mut best_layout = None;
        
        if let Some(layout) = self.try_layout(high, &settings) {
            return layout;
        }

        for _ in 0..10 {
            let mid = (low + high) / 2.0;
            match self.try_layout(mid, &settings) {
                Some(layout) => {
                    best_layout = Some(layout);
                    low = mid;
                }
                None => {
                    high = mid;
                }
            }
        }

        best_layout.unwrap_or_else(|| {
            self.try_layout(min_font, &settings).unwrap_or(LayoutResult {
                words: vec![],
                font_size: min_font,
            })
        })
    }

    pub fn calculate_frame(&self, layout: &LayoutResult, time: f64) -> FrameState {
        let mut active_idx = None;
        
        // Find active word
        for (i, item) in layout.words.iter().enumerate() {
            let word = &self.words[item.word_index];
            if time >= word.start && time < word.end {
                active_idx = Some(i);
                break;
            }
        }
        // Hold last word
        if active_idx.is_none() && !self.words.is_empty() && time >= self.words.last().unwrap().end {
             active_idx = Some(layout.words.len() - 1);
        }

        let mut box_rect = None;

        if let Some(idx) = active_idx {
            let font_size = layout.font_size;
            let target_item = &layout.words[idx];
            let word = &self.words[target_item.word_index];
            
            let box_padding = font_size * 0.6; // Increased padding for safer fit
            let box_h = font_size * 1.4;
            let visual_offset_y = font_size * 0.05;

            // Center box vertically on text middle
            let target_rect = Rect {
                x: target_item.rect.x - box_padding,
                y: (target_item.rect.y + target_item.rect.h / 2.0) - (box_h / 2.0) + visual_offset_y,
                w: target_item.rect.w + (box_padding * 2.0),
                h: box_h,
            };

            let mut current_rect = target_rect;

            // Animation Logic
            if idx > 0 {
                let prev_item = &layout.words[idx - 1];
                let transition_duration = 0.25;
                let time_into_word = time - word.start;

                if time_into_word < transition_duration {
                    let prev_rect = Rect {
                        x: prev_item.rect.x - box_padding,
                        y: (prev_item.rect.y + prev_item.rect.h / 2.0) - (box_h / 2.0) + visual_offset_y,
                        w: prev_item.rect.w + (box_padding * 2.0),
                        h: box_h,
                    };

                    if (prev_rect.y - target_rect.y).abs() > 1.0 {
                        current_rect = target_rect;
                    } else {
                        let t = (time_into_word / transition_duration).clamp(0.0, 1.0);
                        current_rect = interpolate_rect(prev_rect, target_rect, t);
                    }
                }
            }
            
            box_rect = Some(current_rect);
        }

        FrameState {
            active_word_index: active_idx,
            box_rect,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameState {
    pub active_word_index: Option<usize>,
    pub box_rect: Option<Rect>,
}
