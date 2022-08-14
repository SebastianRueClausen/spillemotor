use glam::{Vec4, UVec3, Vec3, Vec2, UVec4, UVec2};
use ash::vk;
use anyhow::Result;

use std::{iter, mem};

use crate::camera::{Camera, CameraUniforms};
use crate::resource::{self, MappedMemory, Buffer, ResourcePool, Res};
use crate::core::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct DirLight {
    pub direction: Vec4,
    pub irradiance: Vec4,
}

impl Default for DirLight {
    fn default() -> Self {
        Self {
            direction: Vec4::new(-1.0, -1.0, 0.0, 1.0).normalize(),
            irradiance: Vec4::splat(0.2),
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit)]
pub struct PointLight {
    world_position: Vec4,
    lum: Vec3,
    // Used to determine the which clusters are effected by the light.
    //
    // TODO: Could perhaps be calculated from `lum`?.
    radius: f32,
}

impl PointLight {
    pub fn new(pos: Vec3, lum: Vec3, radius: f32) -> Self {
        Self {
            world_position: Vec4::from((pos, 1.0)),
            lum,
            radius,
        }
    }
}

#[repr(C)]
struct LightPos {
    view_pos: Vec3,
    radius: f32,
}

/// The data of the light buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct LightBufferData {
    point_light_count: u32, 

    dir_light: DirLight,
    point_lights: [PointLight; MAX_LIGHT_COUNT],
}

unsafe impl bytemuck::NoUninit for LightBufferData {}

impl LightBufferData {
    fn new(lights: &[PointLight]) -> Self {
        let mut point_lights = [PointLight::default(); MAX_LIGHT_COUNT];
        let point_light_count = lights.len() as u32;

        for (src, dst) in lights.iter().zip(point_lights.iter_mut()) {
            *dst = *src;
        }

        Self { point_lights, point_light_count, dir_light: DirLight::default() }
    }
}

#[repr(C)]
struct Aabb {
    min: Vec4,
    max: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct ClusterInfo {
    /// The number of subdivisions in each axis.
    ///
    /// w is the total number of clusters:
    ///
    /// ```ignore
    /// w = divisions.x * divisions.y * divisions.z;
    /// ```
    subdivisions: UVec4,

    /// The size of the clusters in screen space in the x and y dimensions.
    ///
    /// The size on the z-axis is not constant but scales logarithmic as nears the z_far plane.
    cluster_size: UVec2,

    depth_factors: Vec2,
}

impl ClusterInfo {
    fn new(swapchain: &Swapchain, camera: &Camera) -> Self {
        let width = swapchain.extent.width;
        let height = swapchain.extent.height;

        let subdivisions = UVec4::new(12, 12, 24, 12 * 12 * 24);
        let cluster_size = UVec2::new(width / subdivisions.x, height / subdivisions.y);

        let depth_factors = Vec2::new(
            subdivisions.z as f32 / (camera.z_far / camera.z_near).ln(),
            subdivisions.z as f32 * camera.z_near.ln() / (camera.z_far / camera.z_near).ln(),
        );

        Self { subdivisions, cluster_size, depth_factors }
    }

    pub fn cluster_subdivisions(&self) -> UVec3 {
        self.subdivisions.truncate()
    }

    pub fn cluster_count(&self) -> u32 {
        self.subdivisions.w
    }
}

pub struct ClusterInfoBuffer {
    pub buffer: Res<Buffer>,
    mapped: MappedMemory,
    pub info: ClusterInfo,
}

impl ClusterInfoBuffer {
    fn new(renderer: &Renderer, pool: &ResourcePool, camera: &Camera) -> Result<Self> {
        let info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .size(mem::size_of::<ClusterInfo>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let buffer = Buffer::new(&renderer, pool, &info, memory_flags)?;
        let mapped = MappedMemory::new(buffer.block.clone())?;
        let info = ClusterInfo::new(&renderer.swapchain, camera);

        mapped.get_buffer_data(&buffer).copy_from_slice(bytemuck::bytes_of(&info));

        Ok(Self { buffer, mapped, info })
    }

    fn handle_resize(&mut self, camera: &Camera, swapchain: &Swapchain) {
        self.info = ClusterInfo::new(swapchain, camera);
        self.mapped
            .get_buffer_data(&self.buffer)
            .copy_from_slice(bytemuck::bytes_of(&self.info));
    }
}

#[repr(C)]
struct LightMask {
    bits: [u32; MAX_LIGHT_COUNT.div_ceil(32)]
}

/// Pipelines and buffers used to implement clustered shading as described in the paper.
/// "clustered deferred and forward shading - Ola Olsson, Markus Billeter and Ulf Assarsson".
///
/// # The pipeline stages
///
/// Three compute shaders are used to cluster and cull the lights, two of which runs every frame.
///
/// ## Cluster build
/// 
/// This only runs once on startup and every time the window is resized. It's calculates an AABB
/// for each cluster in view space. 
///
/// ## Light update
///
/// This is the first shader to run every frame. It runs in `O(n)` time, where n is the number of
/// lights in the scene. It simply updates the light position buffer by transforming the world
/// space coordinate of each light into view space.
///
/// ## Cluster Update
///
/// This runs after the light update stage is done. It's job is to find which lights effects which
/// clusters. For now it's very naiv. It works by simply iterating through each cluster and
/// checking which lights sphere intersects with the clusters AABB. This means that it has a time
/// complexity of `O(n * k)` time where n is the number of lights and `k` the number of clusters.
///
/// This could be improved by an accelerated structure such as an BVH as descriped in the paper
/// "clustered deferred and forward shading - Ola Olsson, Markus Billeter and Ulf Assarsson".
///
/// # The different buffers
///
/// ## Light buffer
///
/// This is the data of [`LightBufferData`] and contains all the lights in the scene. For now it's
/// static meaning that all the lights are uploaded up front and can't be changed at runtime
/// without temperarely stopping rendering and uploading the new data via a staging buffer.
///
/// ## AABB buffer
///
/// This holds an [`Aabb`] for each cluster in the view fustrum. This one is created at startup and
/// recreated if the resolution changes.
///
/// ## Light mask buffer
///
/// A list of [`LightMask`] for each cluster. This is updated each frame before starting the
/// shading pass. It has one copy per frame in flight.
///
/// This buffer has a copy for each frame in flight and is updated before drawing each frame by
/// the `cluster_update` compute shader.
//
/// This is one of two main ways of doing this. Another way would be to have some kind of light
/// list where each cluster get's it's own slice of light indices. This would work as well, but
/// would take up more a lot more memory unless you start doing heuristics about how many lights
/// each cluster can have, which will show artifacts it that limit is reached.
///
/// ## Light position buffer
///
/// A list of the positions of all lights in view space. This is updated before doing light culling
/// each frame. This is simply an optimization as we could just as easily transform the positions
/// during light culling, but since light culling runs in `O(n * k)` time where n is the number
/// of lights and `k` the number of clusters, this will hopefully speed things up.
///
pub struct Lights {
    pub buffers: Vec<Res<Buffer>>,
    pub light_count: u32,
    pub cluster_info: ClusterInfoBuffer,
    pub cluster_build: ComputeProgram,
    pub light_update: ComputeProgram,
    pub cluster_update: ComputeProgram,
}

impl Lights {
    pub fn new(
        renderer: &Renderer,
        pool: &ResourcePool,
        camera_uniforms: &CameraUniforms,
        camera: &Camera,
        lights: &[PointLight],
    ) -> Result<Self> {
        let cluster_info = ClusterInfoBuffer::new(renderer, pool, camera)?;
        let cluster_count = cluster_info.info.cluster_count() as usize;

        let light_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .size(mem::size_of::<LightBufferData>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let light_position_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size(mem::size_of::<[LightPos; MAX_LIGHT_COUNT]>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let cluster_aabb_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size((cluster_count * mem::size_of::<Aabb>()) as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let light_mask_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size((cluster_count * mem::size_of::<LightMask>()) as u64)
            .build();
        let buffers = {
            let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let mut create_infos = vec![
                light_buffer,
                cluster_aabb_buffer,
            ];

            for info in iter::repeat(light_mask_buffer).take(FRAMES_IN_FLIGHT) {
                create_infos.push(info); 
            }

            for info in iter::repeat(light_position_buffer).take(FRAMES_IN_FLIGHT) {
                create_infos.push(info); 
            }

            let (buffers, _) = resource::create_buffers(&renderer, pool, &create_infos, memory_flags, 4)?;

            buffers
        };

        let light_staging = {
            let info = vk::BufferCreateInfo::builder()
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .size((mem::size_of::<LightBufferData>()) as u64)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let buffer = Buffer::new(&renderer, pool, &info, memory_flags)?;
            let light_data = LightBufferData::new(lights); 

            MappedMemory::new(buffer.block.clone())?
                .get_buffer_data(&buffer)
                .copy_from_slice(bytemuck::bytes_of(&light_data));

            buffer
        };

        renderer.device.transfer_with(|recorder| {
            recorder.copy_buffers(&light_staging, &buffers[0])
        })?;

        let cluster_build = {
            let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
                LayoutBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
                LayoutBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
                LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
            ])?);

            let descriptor = DescriptorSet::new_single(&renderer, layout, &[
                DescriptorBinding::Buffer([cluster_info.buffer.clone()]),
                DescriptorBinding::Buffer([camera_uniforms.proj_uniform().clone()]),
                // Cluster AABB buffer.
                DescriptorBinding::Buffer([buffers[1].clone()]),
            ])?;

            let code = include_bytes_aligned_as!(u32, "../shaders/cluster_build.spv");
            let shader = ShaderModule::new(&renderer, "main", code)?;

            let layout = pool.alloc(
                PipelineLayout::new(&renderer, &[], &[descriptor.layout.clone()])?
            );

            let pipeline = ComputePipeline::new(&renderer, layout, &shader)?;

            ComputeProgram { pipeline, descriptor }
        };

        let light_update = {
            let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
                LayoutBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
                LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
                LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
            ])?);

            let descriptor = DescriptorSet::new_per_frame(&renderer, layout, &[
                DescriptorBinding::Buffer([
                    camera_uniforms.view_uniform(0).clone(),
                    camera_uniforms.view_uniform(1).clone(),
                ]),
                // Light buffer.
                DescriptorBinding::Buffer([buffers[0].clone(), buffers[0].clone()]),
                // Light position buffer.
                DescriptorBinding::Buffer([buffers[4].clone(), buffers[5].clone()]),
            ])?;

            let code = include_bytes_aligned_as!(u32, "../shaders/light_update.spv");
            let shader = ShaderModule::new(&renderer, "main", code)?;

            let layout = pool.alloc(
                PipelineLayout::new(&renderer, &[], &[descriptor.layout.clone()])?
            );

            let pipeline = ComputePipeline::new(&renderer, layout, &shader)?;

            ComputeProgram { pipeline, descriptor }
        };

        let cluster_update = {
            let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
                LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
                LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
                LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
                LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                },
            ])?);

            let descriptor = DescriptorSet::new_per_frame(&renderer, layout, &[
                // Light buffer.
                DescriptorBinding::Buffer([buffers[0].clone(), buffers[0].clone()]),
                // Light position buffer.
                DescriptorBinding::Buffer([buffers[4].clone(), buffers[5].clone()]),
                // AABB buffer.
                DescriptorBinding::Buffer([buffers[1].clone(), buffers[1].clone()]),
                // Light mask buffer.
                DescriptorBinding::Buffer([buffers[2].clone(), buffers[3].clone()]),
            ])?;

            let code = include_bytes_aligned_as!(u32, "../shaders/cluster_update.spv");
            let shader = ShaderModule::new(&renderer, "main", code)?;

            let layout = pool.alloc(
                PipelineLayout::new(&renderer, &[], &[descriptor.layout.clone()])?
            );

            let pipeline = ComputePipeline::new(&renderer, layout, &shader)?;

            ComputeProgram { pipeline, descriptor }
        };

        let light_count = lights.len() as u32;

        let lights = Self {
            buffers,
            cluster_info,
            light_count,
            cluster_build,
            light_update,
            cluster_update,
        };

        lights.build_clusters(renderer)?;

        Ok(lights)
    }

    fn build_clusters(&self, renderer: &Renderer) -> Result<()> {
        renderer.device.transfer_with(|recorder| {
            recorder.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                self.cluster_build.pipeline.layout(),
                &[&self.cluster_build.descriptor],
            );

            let subdivisions = self.cluster_info.info.cluster_subdivisions();
            recorder.dispatch(&self.cluster_build.pipeline, subdivisions);
        })
    }

    pub fn light_buffer(&self) -> &Res<Buffer> {
        &self.buffers[0]
    }

    #[allow(unused)]
    pub fn cluster_aabb_buffer(&self) -> &Res<Buffer> {
        &self.buffers[1]
    }

    pub fn light_mask_buffer(&self, frame: usize) -> &Res<Buffer> {
        &self.buffers[2 + frame]
    }

    pub fn light_position_buffer(&self, frame: usize) -> &Res<Buffer> {
        &self.buffers[4 + frame]
    }

    /// Handle window resize.
    ///
    /// # Warning
    ///
    /// This must only be called when the device is idle, e.g. no rendering is happening, as
    /// during so will upload data to a buffer which might be in use.
    pub fn handle_resize(&mut self, renderer: &Renderer, camera: &Camera) -> Result<()> {
        self.cluster_info.handle_resize(camera, &renderer.swapchain);
        self.build_clusters(renderer)
    }
}

pub struct ComputeProgram {
    pub descriptor: DescriptorSet,
    pub pipeline: ComputePipeline,
}

const MAX_LIGHT_COUNT: usize = 256;
