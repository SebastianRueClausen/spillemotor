use glam::{Vec3, Mat4};
use winit::event::VirtualKeyCode;

use std::time::Duration;

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

pub struct Camera {
    position: Vec3,
    front: Vec3,
    up: Vec3,

    pub perspective: Mat4,
    pub view: Mat4,

    yaw: f32,
    pitch: f32,
    fov: f32,
    znear: f32,
    zfar: f32,

    rotation_speed: f32,
    movement_speed: f32,
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        let position = Vec3::new(10.0, 10.0, 10.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let front = Vec3::default();

        let yaw = 0.0;
        let pitch = 0.0;

        let fov = 66.0_f32;
        let znear = 1.0;
        let zfar = 1000.0;

        let view = Mat4::look_at_rh(position, position, up);
        let perspective = Mat4::perspective_rh(
            fov.to_radians(),
            aspect_ratio,
            znear,
            zfar,
        );

        Self {
            rotation_speed: 0.5,
            movement_speed: 60.0,
            position,
            up,
            front,
            yaw,
            pitch,
            fov,
            znear,
            zfar,
            view,
            perspective,
        }

    }

    pub fn update(&mut self, input_state: &mut InputState, dt: Duration) {
        let speed = self.movement_speed * dt.as_secs_f32();

        if input_state.is_key_pressed(VirtualKeyCode::W) {
            self.position += self.front * speed;
        }
        if input_state.is_key_pressed(VirtualKeyCode::S) {
            self.position -= self.front * speed;
        }
        if input_state.is_key_pressed(VirtualKeyCode::A) {
            self.position -= self.front.cross(self.up).normalize() * speed;
        }
        if input_state.is_key_pressed(VirtualKeyCode::D) {
            self.position += self.front.cross(self.up).normalize() * speed;
        }

        let (x_delta, y_delta) = input_state.mouse_delta();
       
        self.yaw -= x_delta as f32 * self.rotation_speed;
        self.pitch += y_delta as f32 * self.rotation_speed;

        self.pitch = self.pitch.clamp(-89.0, 89.0);

        self.front = -Vec3::new(
            f32::cos(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
            f32::sin(self.pitch.to_radians()),
            f32::sin(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
        )
        .normalize();

        self.view = Mat4::look_at_rh(self.position, self.position + self.front, self.up);
    }

    pub fn update_perspective(&mut self, aspect_ratio: f32) {
        self.perspective = Mat4::perspective_rh(
            self.fov.to_radians(),
            aspect_ratio,
            self.znear,
            self.zfar,
        );
    }
}
