#![feature(let_else)]

#[macro_use]
extern crate log;

#[macro_use]
extern crate anyhow;

#[macro_use]
mod macros;

mod camera;
mod resource;
mod core;
mod util;
mod scene;

use anyhow::Result;

use std::time::{Instant, Duration};

use crate::camera::InputState;
use crate::core::Renderer;

fn main() -> Result<()> {
    env_logger::init();

    use winit::event::{Event, WindowEvent, ElementState};
    use winit::event_loop::ControlFlow;

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    window.set_cursor_grab(false)?;
    window.set_cursor_visible(false);

    let mut minimized = false;

    let mut last_draw = Instant::now();
    let mut last_update = Instant::now();

    let mut renderer = Renderer::new(&window)?;
    let mut input_state = InputState::default();

    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *controlflow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput { input, .. },
            ..
        } => {
            if let Some(key) = input.virtual_keycode {
                match input.state {
                    ElementState::Pressed => input_state.key_pressed(key),
                    ElementState::Released => input_state.key_released(key),
                }
            }
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            input_state.mouse_moved((position.x, position.y));
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            if size.width == 0 && size.height == 0 {
                minimized = true
            } else {
                minimized = false;
                renderer.resize(&window).expect("failed to resize window");
            }
        }
        Event::RedrawRequested(_) => {
            trace!("frame time: {} ms", last_draw.elapsed().as_millis());

            last_draw = Instant::now();

            if !minimized && !renderer.draw().expect("failed drawing to the screen") {
                warn!("out of data surface when trying to draw");
            }
        }
        Event::MainEventsCleared => {
            renderer.camera.update(&mut input_state, last_update.elapsed());
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
