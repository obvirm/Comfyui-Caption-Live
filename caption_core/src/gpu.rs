use wgpu::{Device, Queue, TextureFormat, RenderPass, MultisampleState, TextureDescriptor, TextureDimension, TextureUsages, Extent3d, CommandEncoderDescriptor, RenderPassDescriptor, LoadOp, Operations, BufferDescriptor, BufferUsages, MapMode};
use glyphon::{FontSystem, SwashCache, TextAtlas, TextRenderer, TextArea, Resolution, TextBounds, Attrs, Family, Metrics, Shaping};
use crate::DrawCommand;

pub struct GpuRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
}

impl GpuRenderer {
    pub fn new(device: &Device, queue: &Queue, format: TextureFormat) -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let mut text_atlas = TextAtlas::new(device, queue, format);
        let text_renderer = TextRenderer::new(&mut text_atlas, device, MultisampleState::default(), None);

        Self {
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
        }
    }

    pub fn render<'a>(
        &'a mut self, 
        device: &Device, 
        queue: &Queue, 
        pass: &mut RenderPass<'a>, 
        commands: &[DrawCommand], 
        width: u32, 
        height: u32
    ) {
        let mut text_entries = Vec::new();

        // Process Commands
        for cmd in commands {
            match cmd {
                DrawCommand::DrawRect { rect: _, color: _, radius: _ } => {
                    // TODO: Implement Shape Rendering
                },
                DrawCommand::DrawText { text, x, y, font_size, fill_color, stroke_color: _, stroke_width: _, scale, rotation: _ } => {
                    let mut buffer = glyphon::Buffer::new(&mut self.font_system, Metrics::new(*font_size as f32, *font_size as f32));
                    
                    // Set text with default attributes
                    buffer.set_text(&mut self.font_system, text, Attrs::new().family(Family::SansSerif), Shaping::Advanced);
                    
                    // Shape the buffer
                    buffer.shape_until_scroll(&mut self.font_system);

                    let color = crate::parse_hex_rgba(fill_color);
                    let glyphon_color = glyphon::Color::rgba(color[0], color[1], color[2], color[3]);

                    text_entries.push((buffer, *x, *y, *scale, glyphon_color));
                }
            }
        }
        
        let text_areas: Vec<TextArea> = text_entries.iter().map(|(buffer, x, y, scale, color)| {
            TextArea {
                buffer,
                left: *x as f32,
                top: *y as f32,
                scale: *scale as f32,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                },
                default_color: *color,
            }
        }).collect();
        
        // Prepare and Render Text
        let _ = self.text_renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.text_atlas,
            Resolution { width, height },
            text_areas,
            &mut self.swash_cache,
        );

        let _ = self.text_renderer.render(&self.text_atlas, pass);
    }
}

pub struct HeadlessContext {
    device: Device,
    queue: Queue,
    renderer: GpuRenderer,
    texture: wgpu::Texture,
    output_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl HeadlessContext {
    pub async fn new(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await.expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.expect("Failed to create device");

        let texture_desc = TextureDescriptor {
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_DST,
            label: None,
            view_formats: &[],
        };
        let texture = device.create_texture(&texture_desc);

        // Buffer for reading back data
        let u32_size = std::mem::size_of::<u32>() as u32;
        let output_buffer_size = (u32_size * width * height) as wgpu::BufferAddress;
        let output_buffer_desc = BufferDescriptor {
            size: output_buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

        let renderer = GpuRenderer::new(&device, &queue, TextureFormat::Rgba8UnormSrgb);

        Self {
            device,
            queue,
            renderer,
            texture,
            output_buffer,
            width,
            height,
        }
    }

    pub async fn render_frame(&mut self, image_data: &[u8], commands: &[DrawCommand]) -> Vec<u8> {
        // 1. Upload background image
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width),
                rows_per_image: Some(self.height),
            },
            Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );

        // 2. Render
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load, // Load the background image we just uploaded
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.renderer.render(&self.device, &self.queue, &mut pass, commands, self.width, self.height);
        }

        // 3. Copy to buffer
        let u32_size = std::mem::size_of::<u32>() as u32;
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(u32_size * self.width),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );

        self.queue.submit(Some(encoder.finish()));

        // 4. Read back
        let buffer_slice = self.output_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        self.output_buffer.unmap();

        result
    }
}
