use glam::{Mat4, Vec3A, Vec3, UVec3, UVec2};
use ash::vk;
use anyhow::Result;

use std::{iter, mem};

use crate::camera::Camera;
use crate::resource::{MappedMemory, Buffers, Buffer, BufferView};
use crate::core::{Device, Swapchain, FRAMES_IN_FLIGHT};
use crate::util;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct DirLight {
    /// The direction to the light.
    pub direction: Vec3A,
    pub irradiance: Vec3A,
}

impl Default for DirLight {
    fn default() -> Self {
        Self {
            direction: Vec3A::new(-1.0, -1.0, 0.0).normalize(),
            irradiance: Vec3A::splat(1.0),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub pos: Vec3A,
    pub lum: Vec3A,
    // Used to determine the which clusters are effected by the light.
    pub radius: f32,
}

impl PointLight {
    pub fn new(pos: Vec3, lum: Vec3, radius: f32) -> Self {
        Self {
            pos: Vec3A::from(pos),
            lum: Vec3A::from(lum),
            radius,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
struct LightBufferData {
    count: u32, 
    capacity: u32,
    lights: [PointLight; MAX_LIGHT_COUNT],
}

#[repr(C)]
struct ClusterAabb {
    max: Vec3A,
    min: Vec3A,
}

#[repr(C)]
struct ClusterLights {
    count: u32,
    offset: u32, 
}

#[repr(C)]
pub struct GeneralData {
    inv_perspective: Mat4,
    screen_extent: UVec2,
    cluster_grid: UVec3,
}

unsafe impl util::Pod for GeneralData {}

impl GeneralData {
    pub fn new(camera: &Camera, swapchain: &Swapchain) -> Self {
        Self {
            inv_perspective: camera.perspective.inverse(),
            screen_extent: UVec2::new(
                swapchain.extent.width,
                swapchain.extent.height,
            ),
            cluster_grid: UVec3::new(
                CLUSTER_X as u32,
                CLUSTER_Y as u32,
                CLUSTER_Z as u32,
            ),
        }
    }
}

pub struct GeneralDataBuffer {
    pub buffer: Buffer,
    mapped: MappedMemory,
}

impl GeneralDataBuffer {
    fn new(device: &Device, camera: &Camera, swapchain: &Swapchain) -> Result<Self> {
        let info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size(mem::size_of::<GeneralData>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let buffer = Buffer::new(device, &info, memory_flags)?;
        let mapped = buffer.block.map(device)?;

        let data = GeneralData::new(camera, swapchain);

        unsafe {
            mapped.get_range(..).copy_from_slice(util::bytes_of(&data));
        }

        Ok(Self { buffer, mapped })

    }

    fn handle_resize(&self, camera: &Camera, swapchain: &Swapchain) {
        let data = GeneralData::new(camera, swapchain);
        unsafe {
            self.mapped.get_range(..).copy_from_slice(util::bytes_of(&data));
        }
    }

    pub unsafe fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
    }
}

/// # The different buffers
///
/// ## Light buffer
///
/// This is the data of [`LightBufferData`] and contains all the lights in the scene. For now it's
/// static meaning that all the lights are uploaded up front and can't be changed at runtime
/// without temperarely stopping rendering and uploading the new data via a staging buffer.
///
/// however it could become dynamic in the future, in which case we will probably have to have a
/// copy per frame in flight.
///
/// Visible in the fragment shader.
///
/// ## Cluster AABB buffer
///
/// This holds a [`ClusterAabb`] for each cluster in the view fustrum. There is one copy per frame in
/// flight.
///
/// ## Active light buffer
///
/// A simple list of all the indices of lights that are active and effects a cluster. There is one
/// copy per frame in flight. Visible in the fragment shader.
///
/// ## Cluster light buffer
///
/// This holds a [`ClusterLights`] for each cluster. There is one copy per frame in flight. Visible
/// to the fragment shader.
///
pub struct Lights {
    buffers: Buffers,
    pub general_data_buffer: GeneralDataBuffer,
}

impl Lights {
    pub fn new(
        device: &Device,
        camera: &Camera,
        swapchain: &Swapchain,
        lights: &[PointLight],
    ) -> Result<Self> {
        let light_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .size(mem::size_of::<LightBufferData>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let cluster_aabb_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size((CLUSTER_COUNT * mem::size_of::<ClusterAabb>()) as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let active_lights_buffer = vk::BufferCreateInfo::builder() .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size((LIGHTS_PER_CLUSTER * CLUSTER_COUNT * mem::size_of::<u32>()) as u64)
            .build();
        let cluster_lights_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size((CLUSTER_COUNT * mem::size_of::<ClusterLights>()) as u64)
            .build();

        let buffers = {
            let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let mut create_infos = vec![light_buffer];

            for info in iter::repeat(cluster_aabb_buffer).take(FRAMES_IN_FLIGHT) {
                create_infos.push(info); 
            }

            for info in iter::repeat(active_lights_buffer).take(FRAMES_IN_FLIGHT) {
                create_infos.push(info); 
            }

            for info in iter::repeat(cluster_lights_buffer).take(FRAMES_IN_FLIGHT) {
                create_infos.push(info); 
            }

            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        let light_staging = {
            let info = vk::BufferCreateInfo::builder()
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .size((mem::size_of::<LightBufferData>()) as u64)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let buffer = Buffer::new(device, &info, memory_flags)?;

            buffer.block.map_with(device, |mapped| unsafe {
                let ptr: *mut LightBufferData = mapped
                    .as_ptr()
                    .expect("light staging buffer doesn't have correct aligment");

                (*ptr).count = lights.len() as u32;
                (*ptr).capacity = MAX_LIGHT_COUNT as u32;

                for (dst, src) in (*ptr).lights.iter_mut().zip(lights.iter()) {
                    *dst = src.clone();
                }
            })?;

            buffer
        };

        device.transfer_with(|command_buffer| unsafe {
            let light_buffer = &buffers[0];
            assert_eq!(light_buffer.size(), light_staging.size());
            let regions = [vk::BufferCopy::builder()
                .src_offset(0)
                .dst_offset(0)
                .size(light_staging.size())
                .build()];
            device.handle.cmd_copy_buffer(
                command_buffer,
                light_staging.handle,
                light_buffer.handle,
                &regions,
            );
        })?;

        unsafe { light_staging.destroy(device); }

        let general_data_buffer = GeneralDataBuffer::new(device, camera, swapchain)?;

        Ok(Self { buffers, general_data_buffer })
    }

    pub fn light_buffer(&self) -> &BufferView {
        &self.buffers[0]
    }

    pub fn cluster_aabb_buffer(&self, frame: usize) -> &BufferView {
        &self.buffers[1 + frame]
    }

    pub fn active_lights_buffer(&self, frame: usize) -> &BufferView {
        &self.buffers[3 + frame]
    }

    pub fn cluster_lights_buffer(&self, frame: usize) -> &BufferView {
        &self.buffers[5 + frame]
    }

    /// Handle window resize.
    ///
    /// # Warning
    ///
    /// This must only be called when the device is idle, e.g. not rendering is happening, as
    /// during so will upload data to a buffer which might be in use.
    pub fn handle_resize(&mut self, camera: &Camera, swapchain: &Swapchain) {
        self.general_data_buffer.handle_resize(camera, swapchain);
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.buffers.destroy(device);
        self.general_data_buffer.destroy(device);
    }
}

const CLUSTER_X: usize = 8;
const CLUSTER_Y: usize = 4;
const CLUSTER_Z: usize = 8;

const CLUSTER_COUNT: usize = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
const LIGHTS_PER_CLUSTER: usize = 8;
const MAX_LIGHT_COUNT: usize = 128;

