use glam::{Vec4, UVec3, Vec3, Vec2, UVec4, UVec2};
use ash::vk;
use anyhow::Result;

use std::{iter, mem};

use crate::camera::Camera;
use crate::resource::{MappedMemory, Buffers, Buffer, BufferView};
use crate::core::{
    Device,
    Swapchain,
    Pipeline,
    PipelineRequest,
    ShaderModule,
    FRAMES_IN_FLIGHT,
    CameraUniforms,
    DescriptorSet,
    DescriptorBinding,
    BindingKind,
    DescriptorLayoutCache,
};

#[repr(C)]
#[derive(Debug, Clone)]
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

/// SPEED! SPEED! SPEED!
///
/// Split this into two parts:
///
/// - Cull: The radius and position in view space.
/// - Draw: The lumen and position in world space.
///
/// Then we can run the update light compute shader which updates the view position on the cull
/// struct. This shader has to access both the cull and draw parts, but only visits each light
/// once, i.e. O(n).
///
/// When running the cluster update shader, we only have to access the cull data, and when drawing
/// we only have to access the draw data.
///
/// We will have to have multiple copies of the cull data, but not draw data. This doesn't take up
/// more vram than having the light position buffer we have now, since the cull data can fit into a
/// single vec4.
///
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub world_position: Vec4,
    pub lum: Vec4,
    // Used to determine the which clusters are effected by the light.
    //
    // TODO: Could perhaps be calculated from `lum`?.
    pub radius: f32,
}

impl PointLight {
    pub fn new(pos: Vec3, lum: Vec3, radius: f32) -> Self {
        Self {
            world_position: Vec4::from((pos, 1.0)),
            lum: Vec4::from((lum, 1.0)),
            radius,
        }
    }
}

/// The data of the light buffer.
#[repr(C)]
#[derive(Debug)]
struct LightBufferData {
    point_light_count: u32, 

    dir_light: DirLight,
    point_lights: [PointLight; MAX_LIGHT_COUNT],
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
    pub buffer: Buffer,
    mapped: MappedMemory,
    pub info: ClusterInfo,
}

impl ClusterInfoBuffer {
    fn new(device: &Device, camera: &Camera, swapchain: &Swapchain) -> Result<Self> {
        let info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .size(mem::size_of::<ClusterInfo>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let buffer = Buffer::new(device, &info, memory_flags)?;
        let mapped = buffer.block.map(device)?;
        let info = ClusterInfo::new(swapchain, camera);

        unsafe { mapped.get_range(..).copy_from_slice(bytemuck::bytes_of(&info)); }

        Ok(Self { buffer, mapped, info })
    }

    fn handle_resize(&mut self, camera: &Camera, swapchain: &Swapchain) {
        self.info = ClusterInfo::new(swapchain, camera);
         
        unsafe {
            self.mapped.get_range(..).copy_from_slice(bytemuck::bytes_of(&self.info));
        }
    }

    pub unsafe fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
    }
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
/// ## Light index buffer
///
/// A list of indices to lights in each cluster. Each cluster has it's own slice of
/// `LIGHTS_PER_CLUSTER` indices. The offset into the list for cluster with index k is calculated
/// as `LIGHTS_PER_CLUSTER * k`. If less than `LIGHTS_PER_CLUSTER` lights are in the cluster, then
/// the end is marked with an terminal invalid index.
///
/// This buffer has a copy for each frame in flight and is updated before drawing each frame by
/// the `cluster_update` compute shader.
//
/// This is one of two main ways of doing this. Another way would be to have another buffer with
/// info for each cluster, with the base offset into this buffer and the amount of lights. It could
/// have some advantages if we add support for another type of light, like flood lights.
///
/// It could perhaps also have some performance improvements as the indices would be denser in
/// memory, but that would also add one more layer of indirection in the fragment shader.
///
/// ## Light position buffer
///
/// A list of the positions of all lights in view space. This is updated before doing light culling
/// each frame. This is simply an optimization as we could just as easily transform the positions
/// during light culling, but since light culling runs in `O(n * k)` time where n is the number
/// of lights and `k` the number of clusters, this will hopefully speed things up by reducing the
/// complexity to `O(n)` but adding an extra memory lookup during culling.
///
pub struct Lights {
    pub buffers: Buffers,
    pub light_count: u32,
    pub cluster_info: ClusterInfoBuffer,
    pub cluster_build: ComputeProgram,
    pub light_update: ComputeProgram,
    pub cluster_update: ComputeProgram,
}

impl Lights {
    pub fn new(
        device: &Device,
        layout_cache: &mut DescriptorLayoutCache,
        camera_uniforms: &CameraUniforms,
        camera: &Camera,
        swapchain: &Swapchain,
        lights: &[PointLight],
    ) -> Result<Self> {
        let cluster_info = ClusterInfoBuffer::new(device, camera, swapchain)?;
        let cluster_count = cluster_info.info.cluster_count() as usize;

        let light_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .size(mem::size_of::<LightBufferData>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let light_position_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size(mem::size_of::<[Vec4; MAX_LIGHT_COUNT]>() as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let cluster_aabb_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .size((cluster_count * mem::size_of::<Aabb>()) as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let light_index_buffer = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size((cluster_count * LIGHTS_PER_CLUSTER * mem::size_of::<u32>()) as u64)
            .build();

        let buffers = {
            let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let mut create_infos = vec![
                light_buffer,
                cluster_aabb_buffer,
            ];
            for info in iter::repeat(light_index_buffer).take(FRAMES_IN_FLIGHT) {
                create_infos.push(info); 
            }
            for info in iter::repeat(light_position_buffer).take(FRAMES_IN_FLIGHT) {
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

                (*ptr).point_light_count = lights.len() as u32;
                (*ptr).dir_light = DirLight::default();

                for (dst, src) in (*ptr).point_lights.iter_mut().zip(lights.iter()) {
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

        let cluster_build = {
            let descriptor = DescriptorSet::new(device, layout_cache, &[
                DescriptorBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&cluster_info.buffer]),
                },
                DescriptorBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([
                        camera_uniforms.proj_uniform(),
                    ]),
                },
                // Cluster AABB buffer.
                DescriptorBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&buffers[1]]),
                },
            ])?;

            let code = include_bytes_aligned_as!(u32, "../shaders/cluster_build.spv");
            let shader = ShaderModule::new(device, "main", code)?;

            let pipeline = Pipeline::new(device, PipelineRequest::Compute {
                descriptors: &[&descriptor], 
                push_constants: &[],
                shader: &shader,
            })?;

            unsafe { shader.destroy(device); }

            ComputeProgram { pipeline, descriptor }
        };

        let light_update = {
            let descriptor = DescriptorSet::new(device, layout_cache, &[
                DescriptorBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([
                        camera_uniforms.view_uniform(0),
                        camera_uniforms.view_uniform(1),
                    ]),
                },
                // Light buffer.
                DescriptorBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&buffers[0], &buffers[0]]),
                },
                // Light position buffer.
                DescriptorBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&buffers[4], &buffers[5]]),
                },
            ])?;

            let code = include_bytes_aligned_as!(u32, "../shaders/light_update.spv");
            let shader = ShaderModule::new(device, "main", code)?;

            let pipeline = Pipeline::new(device, PipelineRequest::Compute {
                descriptors: &[&descriptor], 
                push_constants: &[],
                shader: &shader,
            })?;

            unsafe { shader.destroy(device); }

            ComputeProgram { pipeline, descriptor }
        };

        let cluster_update = {
            let descriptor = DescriptorSet::new(device, layout_cache, &[
                // Light buffer.
                DescriptorBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&buffers[0], &buffers[0]]),
                },
                // Light position buffer.
                DescriptorBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&buffers[4], &buffers[5]]),
                },
                // AABB buffer.
                DescriptorBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&buffers[1], &buffers[1]]),
                },
                // Light index buffer.
                DescriptorBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage: vk::ShaderStageFlags::COMPUTE,
                    kind: BindingKind::Buffer([&buffers[2], &buffers[3]]),
                },
            ])?;

            let code = include_bytes_aligned_as!(u32, "../shaders/cluster_update.spv");
            let shader = ShaderModule::new(device, "main", code)?;

            let pipeline = Pipeline::new(device, PipelineRequest::Compute {
                descriptors: &[&descriptor], 
                push_constants: &[],
                shader: &shader,
            })?;

            unsafe { shader.destroy(device); }

            ComputeProgram { pipeline, descriptor }
        };

        let lights = Self {
            buffers,
            cluster_info,
            light_count: lights.len() as u32,
            cluster_build,
            light_update,
            cluster_update,
        };

        lights.build_clusters(device)?;

        Ok(lights)
    }

    fn build_clusters(&self, device: &Device) -> Result<()> {
        device.transfer_with(|command_buffer| unsafe {
            device.handle.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.cluster_build.pipeline.handle,
            );

            device.handle.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.cluster_build.pipeline.layout,
                0,
                &[self.cluster_build.descriptor[0]],
                &[],
            );

            let subdivisions = self.cluster_info.info.cluster_subdivisions();

            device.handle.cmd_dispatch(
                command_buffer,
                subdivisions.x,
                subdivisions.y,
                subdivisions.z,
            );
        })
    }

    pub fn light_buffer(&self) -> &BufferView {
        &self.buffers[0]
    }

    #[allow(unused)]
    pub fn cluster_aabb_buffer(&self) -> &BufferView {
        &self.buffers[1]
    }

    pub fn light_index_buffer(&self, frame: usize) -> &BufferView {
        &self.buffers[2 + frame]
    }

    pub fn light_position_buffer(&self, frame: usize) -> &BufferView {
        &self.buffers[4 + frame]
    }

    /// Handle window resize.
    ///
    /// # Warning
    ///
    /// This must only be called when the device is idle, e.g. no rendering is happening, as
    /// during so will upload data to a buffer which might be in use.
    pub fn handle_resize(&mut self, device: &Device, camera: &Camera, swapchain: &Swapchain) -> Result<()> {
        self.cluster_info.handle_resize(camera, swapchain);
        self.build_clusters(device)
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.buffers.destroy(device);
        self.cluster_info.destroy(device);
        self.cluster_build.destroy(device);
        self.light_update.destroy(device);
        self.cluster_update.destroy(device);
    }
}

pub struct ComputeProgram {
    pub descriptor: DescriptorSet,
    pub pipeline: Pipeline,
}

impl ComputeProgram {
    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.descriptor.destroy(device);
        self.pipeline.destroy(device);
    }
}

const LIGHTS_PER_CLUSTER: usize = 64;
const MAX_LIGHT_COUNT: usize = 256;
