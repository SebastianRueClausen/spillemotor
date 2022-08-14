#![feature(let_else, int_roundings)]

#[macro_use]
extern crate log;

#[macro_use]
extern crate anyhow;

#[macro_use]
mod macros;

mod resource;
mod core;
mod scene;
mod light;
mod text;
mod camera;

mod gltf_import;
mod font_import;

use anyhow::Result;
use glam::{UVec3, Vec3};
use winit::event::VirtualKeyCode;
use ash::vk;

use std::time::{Instant, Duration};
use std::path::Path;

use crate::core::Renderer;
use crate::text::TextPass;
use crate::scene::Scene;
use crate::light::{Lights, PointLight};
use crate::camera::{Camera, CameraUniforms};
use crate::resource::ResourcePool;

fn main() -> Result<()> {
    env_logger::init();

    use winit::event::{Event, WindowEvent, ElementState};
    use winit::event_loop::ControlFlow;

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    window.set_cursor_grab(false)?;
    window.set_cursor_visible(false);

    let mut minimized = false;

    let mut last_update = Instant::now();
    let mut last_draw = Instant::now();

    let mut renderer = Renderer::new(&window)?;
    let mut input_state = InputState::default();

    let resource_pool = ResourcePool::new();

    let font = font_import::FontData::new(Path::new("fonts/source_code_pro/metadata.json"))?;

    let mut text_pass = TextPass::new(&renderer, &resource_pool, &font)?;

    let mut camera = Camera::new(renderer.swapchain.aspect_ratio());
    let camera_uniforms = CameraUniforms::new(&renderer, &resource_pool, &camera)?;

    let lights = debug_lights();
    let mut lights = Lights::new(&renderer, &resource_pool, &camera_uniforms, &camera, &lights)?;

    let scene_data = gltf_import::load(Path::new("models/sponza/Sponza.gltf"))?;
    let scene = Scene::from_scene_data(&renderer, &resource_pool, &camera_uniforms, &lights, &scene_data)?;

    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            *controlflow = ControlFlow::Exit;
        }
        Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
            if let Some(key) = input.virtual_keycode {
                match input.state {
                    ElementState::Pressed => input_state.key_pressed(key),
                    ElementState::Released => input_state.key_released(key),
                }
            }
        }
        Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
            input_state.mouse_moved((position.x, position.y));
        }
        Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
            if size.width == 0 && size.height == 0 {
                minimized = true
            } else {
                minimized = false;
                renderer.resize(&window).expect("failed to resize window");
                text_pass.handle_resize(&renderer);
                camera.update_proj(renderer.swapchain.aspect_ratio());
                camera_uniforms.update_proj(&renderer, &camera);
                lights.handle_resize(&renderer, &camera).expect("failed to resize lights");
            }
        }
        Event::RedrawRequested(_) => {
            if !minimized {
                let elapsed = last_draw.elapsed();
                last_draw = Instant::now();

                let res = renderer.draw(
                    |recorder| {
                        camera_uniforms.update_view(recorder.frame_index(), &camera);

                        recorder.bind_descriptor_sets(
                            vk::PipelineBindPoint::COMPUTE,
                            lights.light_update.pipeline.layout(),
                            &[&lights.light_update.descriptor],
                        );

                        let group_count = UVec3::new(lights.light_count.div_ceil(64), 1, 1);
                        recorder.dispatch(&lights.light_update.pipeline, group_count);
               
                        recorder.buffer_rw_barrier(
                            lights.light_position_buffer(recorder.frame_index()),
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                        );

                        recorder.bind_descriptor_sets(
                            vk::PipelineBindPoint::COMPUTE,
                            lights.cluster_update.pipeline.layout(),
                            &[&lights.cluster_update.descriptor],
                        );

                        let group_count = lights.cluster_info.info.cluster_subdivisions();
                        recorder.dispatch(&lights.cluster_update.pipeline, group_count);
               
                        recorder.buffer_rw_barrier(
                            lights.light_mask_buffer(recorder.frame_index()),
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::FRAGMENT_SHADER,
                        );
                    },
                    |recorder| {
                        recorder.bind_index_buffer(scene.index_buffer());
                        recorder.bind_vertex_buffer(scene.vertex_buffer());

                        recorder.bind_graphics_pipeline(&scene.render_pipeline);

                        for model in scene.models.iter() {
                            let mat = &scene.materials[model.material];
                            let descs = [&mat.descriptor, &scene.light_descriptor];

                            recorder.bind_descriptor_sets(
                                vk::PipelineBindPoint::GRAPHICS,
                                scene.render_pipeline.layout(),
                                &descs,
                            );
                           
                            recorder.push_constants(
                                &scene.render_pipeline.layout(),
                                vk::ShaderStageFlags::VERTEX,
                                0,
                                *&model.transform(),
                            );

                            recorder.draw(
                                model.index_count,
                                model.index_start,
                                model.vertex_start as i32,
                            );
                        }
                
                        text_pass.draw_text(recorder, |texts| {
                            let fps = format!("fps: {}", 1000 / elapsed.as_millis());
                            texts.add_label(40.0, Vec3::new(20.0, 20.0, 0.5), &fps); 
                        });
                    },
                );

                res.expect("failed rendering");
            }
        }
        Event::MainEventsCleared => {
            camera.update(&mut input_state, last_update.elapsed());
            last_update = Instant::now();

            if let Some(left) = Duration::from_millis(16).checked_sub(last_draw.elapsed()) {
                trace!("sleep for {left:?}");
                *controlflow = ControlFlow::WaitUntil(Instant::now() + left);
            } else {
                window.request_redraw();
            }
        }
        _ => {}
    });
}

#[derive(Default)]
pub struct InputState {
    /// Keeps track of if each `VirtualKeyCode` is pressed or not. Each key code represents a
    /// single bit.
    key_pressed: [u64; 3],
    /// The current position of the mouse. `None` if no `mouse_moved` event has been received.
    mouse_pos: Option<(f64, f64)>,
    /// Contains the mouse position delta since last time `mouse_delta`.
    mouse_delta: Option<(f64, f64)>,
}

impl InputState {
    pub fn mouse_moved(&mut self, pos: (f64, f64)) {
        let mouse_pos = self.mouse_pos.unwrap_or(pos);
        let mouse_delta = self.mouse_delta.unwrap_or((0.0, 0.0));

        self.mouse_delta = Some((
            mouse_delta.0 + (mouse_pos.0 - pos.0),
            mouse_delta.1 + (mouse_pos.1 - pos.1),
        ));

        self.mouse_pos = Some(pos);   
    }

    pub fn key_pressed(&mut self, key: VirtualKeyCode) {
        let major = key as usize / 64;
        let minor = key as usize % 64;

        self.key_pressed[major] |= 1 << minor;
    }

    pub fn key_released(&mut self, key: VirtualKeyCode) {
        let major = key as usize / 64;
        let minor = key as usize % 64;
    
        self.key_pressed[major] &= !(1 << minor);
    }

    pub fn is_key_pressed(&self, key: VirtualKeyCode) -> bool {
        let major = key as usize / 64;
        let minor = key as usize % 64;
   
        self.key_pressed[major] & (1 << minor) != 0
    }

    pub fn mouse_delta(&mut self) -> (f64, f64) {
        self.mouse_delta.take().unwrap_or((0.0, 0.0))
    }
}

fn debug_lights() -> Vec<PointLight> {
    let mut lights = Vec::default();

    for i in 0..20 {
        let red = (i % 2) as f32;
        let blue = ((i + 1) % 2) as f32;

        let start = Vec3::new(-16.0, -3.0, -8.0);
        let end = Vec3::new(15.0, 13.0, 8.0);

        let position = start.lerp(end, i as f32 / 20.0);

        lights.push(PointLight::new(
            position,
            Vec3::new(red, 1.0, blue) * 6.0,
            8.0,
        ));
    }

    lights 
}
