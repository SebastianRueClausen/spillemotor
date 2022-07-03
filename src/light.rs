use glam::{Mat4, Vec3, UVec3, UVec2};
use ash::vk;
use anyhow::Result;

use std::{iter, mem};

use crate::resource::{MappedMemory, Buffers};
use crate::core::{Device, FRAMES_IN_FLIGHT};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct DirLight {
    /// The direction to the light.
    pub direction: Vec3,
    pub irradiance: Vec3,
}

impl Default for DirLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-1.0, -1.0, 0.0).normalize(),
            irradiance: Vec3::splat(1.0),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PointLight {
    pub pos: Vec3,
    pub lum: Vec3,
}

#[repr(C)]
struct ClusterAabb {
    max: Vec3,
    min: Vec3,
}

#[repr(C)]
struct ClusterLights {
    count: u32,
    offset: u32, 
}

#[repr(C)]
struct ClusterAssign {
    /// The inverse of the projection matrix.
    view_to_world: Mat4,
    screen_dim: UVec2,
    cluster_dim: UVec3,
}

pub struct LightBuffers {
    device_buffers: Buffers,
    mapped_buffers: Buffers,
    mapped_memory: MappedMemory,
}

impl LightBuffers {
    pub fn new(device: &Device, lights: &[PointLight]) -> Result<Self> {
        let cluster_assign_info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size(mem::size_of::<ClusterAssign>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let light_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size((MAX_LIGHT_COUNT * mem::size_of::<PointLight>()) as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let cluster_aabb_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size((CLUSTER_COUNT * mem::size_of::<ClusterAabb>()) as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let active_lights_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size((LIGHTS_PER_CLUSTER * CLUSTER_COUNT * mem::size_of::<u32>()) as u64)
            .build();
        let cluster_lights_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size((CLUSTER_COUNT * mem::size_of::<ClusterLights>()) as u64)
            .build();

        let device_buffers = {
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

        let mapped_buffers = {
            let memory_flags
                = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let mut create_infos = Vec::default();
            for info in iter::repeat(cluster_assign_info).take(FRAMES_IN_FLIGHT) {
                create_infos.push(info);
            }

            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        let mapped_memory = mapped_buffers.block.map(device)?;
            
        Ok(Self {
            device_buffers,
            mapped_buffers,
            mapped_memory,
        })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.device_buffers.destroy(device);
        self.mapped_buffers.destroy(device);
    }
}

const CLUSTER_X: usize = 8;
const CLUSTER_Y: usize = 4;
const CLUSTER_Z: usize = 8;

const CLUSTER_COUNT: usize = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
const LIGHTS_PER_CLUSTER: usize = 8;
const MAX_LIGHT_COUNT: usize = 128;

