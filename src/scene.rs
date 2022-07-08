use glam::{Vec3, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::mem;
use std::ops::Index;
use std::time::Duration;

use crate::util;
use crate::camera::{InputState, Camera};
use crate::light::{PointLight, Lights};
use crate::core::{Device, Swapchain, RenderPass, RenderTargets, Pipeline};
use crate::core::{DescriptorSet, DescriptorBinding, BindingKind};
use crate::core::{VertUniform, FragUniform, UniformBuffers};
use crate::resource::{Images, Buffers};

#[repr(C)]
#[derive(Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoord: Vec2,
}

pub struct Model {
    pub index_buffer: usize,
    pub vertex_buffer: usize,  
    pub index_count: u32,
    pub material: usize,
    transform: Mat4,
}

impl Model {
    /// Get byte slice of transform data to transfer to the GPU.
    pub fn transform_data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                &self.transform as *const Mat4 as *const u8,
                mem::size_of::<Mat4>(),
            )
        }
    }
}

pub struct Models {
    pub buffers: Buffers,
    models: Vec<Model>,
}

impl Models {
    pub fn iter(&self) -> impl Iterator<Item = &Model> {
        self.models.iter()
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.buffers.destroy(device);
    }
}

impl Index<usize> for Models {
    type Output = Model;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.models[idx]
    }
}

#[repr(C)]
pub struct MaterialParams {
    pub roughness: f32,
    pub metallic: f32,
}

unsafe impl util::Pod for MaterialParams {}

pub struct Material {
    pub base_color: usize,
    pub pipeline: Pipeline,
    pub descriptor: DescriptorSet,
    pub params: MaterialParams,
}

impl Material {
    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.pipeline.destroy(device);
        self.descriptor.destroy(device);
    }
}

pub struct Materials {
    pub images: Images, 
    pub sampler: vk::Sampler,
    materials: Vec<Material>,
}

impl Index<usize> for Materials {
    type Output = Material;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.materials[idx]
    }
}

impl Materials {
    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        device.handle.destroy_sampler(self.sampler, None);
        self.images.destroy(device);
        for mat in &self.materials {
            mat.destroy(device);
        }
    }
}

pub struct Scene {
    lights: Lights,
    camera: Camera,

    vert_uniform: VertUniform,
    frag_uniform: FragUniform,
    uniform_buffers: UniformBuffers,

    pub light_descriptor: DescriptorSet,

    pub materials: Materials,
    pub models: Models,
}

impl Scene {
    pub fn from_scene_data(
        device: &Device,
        swapchain: &Swapchain,
        render_pass: &RenderPass,
        render_targets: &RenderTargets,
        scene_data: &SceneData,
    ) -> Result<Self> {
        let camera = Camera::new(swapchain.aspect_ratio());
        let lights = Lights::new(device, &camera, &swapchain, &[
            PointLight::new(
                Vec3::new(110.0, 45.0, -24.0),
                Vec3::splat(32000.0),
                10.0,
            ),
            PointLight::new(
                Vec3::new(30.0, 55.0, -59.0),
                Vec3::splat(32000.0),
                10.0,
            ),
        ])?;

        let uniform_buffers = UniformBuffers::new(&device)?;
        let vert_uniform = VertUniform::default();
        let frag_uniform = FragUniform::default();

        let staging = {
            let create_infos: Vec<_> = scene_data.meshes
                .iter()
                .flat_map(|mesh| [
                    vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .size(mesh.verts.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                    vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .size(mesh.indices.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                ])
                .collect();

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        let mapped = staging.block.map(device)?;

        scene_data.meshes.iter()
            .flat_map(|mesh| [mesh.verts.as_slice(), mesh.indices.as_slice()])
            .zip(staging.buffers.iter())
            .for_each(|(src, dst)| unsafe {
                mapped.get_range(dst.range.clone()).copy_from_slice(&src);
            });

        staging.block.unmap(device, mapped);

        let buffers = {
            let create_infos: Vec<_> = scene_data.meshes
                .iter()
                .flat_map(|mesh| [
                    vk::BufferCreateInfo::builder()
                        .usage(
                            vk::BufferUsageFlags::VERTEX_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_DST
                        )
                        .size(mesh.verts.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                    vk::BufferCreateInfo::builder()
                        .usage(
                            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
                        )
                        .size(mesh.indices.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                ])
                .collect();

            Buffers::new(device, &create_infos, vk::MemoryPropertyFlags::DEVICE_LOCAL, 4)?
        };

        device.transfer_with(|command_buffer| {
            for (src, dst) in staging.iter().zip(buffers.iter()) {
                assert_eq!(src.size(), dst.size());

                let regions = [vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(src.size())
                    .build()];

                unsafe {
                    device.handle.cmd_copy_buffer(
                        command_buffer,
                        src.handle,
                        dst.handle,
                        &regions,
                    );
                }
            }
        })?;

        unsafe { staging.destroy(device); }

        let staging = {
            let create_infos: Vec<_> = scene_data.materials
                .iter()
                .flat_map(|mat| [
                    vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .size(mat.base_color.data.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                ])
                .collect();
            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        let mapped = staging.block.map(device)?;

        scene_data.materials
            .iter()
            .map(|mat| mat.base_color.data.as_slice())
            .zip(staging.iter())
            .for_each(|(src, dst)| unsafe {
                mapped.get_range(dst.range.clone()).copy_from_slice(&src);
            });

        staging.block.unmap(device, mapped);

        let mut images = {
            let image_infos: Vec<_> = scene_data.materials
                .iter()
                .map(|mat| {
                    vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                        .format(vk::Format::R8G8B8A8_SRGB)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .extent(vk::Extent3D {
                            width: mat.base_color.width,
                            height: mat.base_color.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .build()
                })
                .collect();

            let view_infos: Vec<_> = scene_data.materials
                .iter()
                .map(|_| {
                    let subresource_range = vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build();
                    vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_SRGB)
                        .subresource_range(subresource_range)
                        .build()
                })
                .collect();

            Images::new(device, &image_infos, &view_infos, vk::MemoryPropertyFlags::DEVICE_LOCAL)?
        };

        for image in images.images.iter_mut() {
            image.transition_layout(device, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        }

        device.transfer_with(|command_buffer| {
            for (src, dst) in staging.iter().zip(images.iter()) {
                // I think they should be the same size, but in some instance they aren't for some
                // reason. Vulkan doesn't complain so i guess it's is alright.
                assert!(src.size() <= dst.size());

                let subresource = vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build();
                let regions = [vk::BufferImageCopy::builder()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_extent(dst.extent)
                    .image_subresource(subresource)
                    .build()];
                unsafe {
                    device.handle.cmd_copy_buffer_to_image(
                        command_buffer,
                        src.handle,
                        dst.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    );
                }
            }
        })?;

        unsafe { staging.destroy(device); }

        for image in images.images.iter_mut() {
            image.transition_layout(device, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        }

        let models: Vec<_> = scene_data.meshes
            .iter()
            .enumerate()
            .map(|(i, mesh)| {
                Model {
                    material: mesh.material,
                    transform: mesh.transform,
                    vertex_buffer: i * 2,
                    index_buffer: i * 2 + 1,
                    index_count: (mesh.indices.len() / mem::size_of::<u16>()) as u32,
                }
            })
            .collect();

        let light_descriptor = DescriptorSet::new(device, &[
            DescriptorBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                kind: BindingKind::Buffer([
                    &lights.general_data_buffer.buffer, 
                    &lights.general_data_buffer.buffer, 
                ]),
            },
            DescriptorBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                kind: BindingKind::Buffer([
                    lights.light_buffer(), 
                    lights.light_buffer(), 
                ]),
            },
            DescriptorBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                kind: BindingKind::Buffer([
                    lights.active_lights_buffer(0), 
                    lights.active_lights_buffer(1), 
                ]),
            },
            DescriptorBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                kind: BindingKind::Buffer([
                    lights.cluster_lights_buffer(0), 
                    lights.cluster_lights_buffer(1), 
                ]),
            },
        ])?;

        let sampler = create_texture_sampler(device)?;
        let materials: Result<Vec<_>> = scene_data.materials
            .iter()
            .enumerate()
            .map(|(base_color, mat)| {
                let base = &images[base_color];
                let descriptor = DescriptorSet::new(device, &[
                    DescriptorBinding {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        stage: vk::ShaderStageFlags::VERTEX,
                        kind: BindingKind::Buffer([
                            uniform_buffers.get_vert(0),
                            uniform_buffers.get_vert(1),
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Buffer([
                            uniform_buffers.get_frag(0),
                            uniform_buffers.get_frag(1),
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Image(sampler, [base; 2]),
                    },
                ])?;

                let pipeline = Pipeline::new_render(
                    device,
                    swapchain,
                    render_pass,
                    render_targets,
                    &[&descriptor, &light_descriptor],
                )?;

                Ok(Material {
                    base_color,
                    pipeline,
                    descriptor,
                    params: MaterialParams {
                        metallic: mat.metallic,
                        roughness: mat.roughness,
                    },
                })
            })
            .collect();

        let materials = materials?;

        Ok(Self {
            camera,
            lights,
            light_descriptor,
            vert_uniform,
            frag_uniform,
            uniform_buffers,
            models: Models {
                buffers,
                models,
            },
            materials: Materials {
                images,
                sampler,
                materials,
            },
        })
    }

    pub fn update(&mut self, input_state: &mut InputState, dt: Duration) {
        self.camera.update(input_state, dt); 
    }

    pub fn handle_resize(&mut self, swapchain: &Swapchain) {
        self.camera.update_perspective(swapchain.aspect_ratio());
        self.lights.handle_resize(&self.camera, swapchain); 
    }

    pub fn upload_data(&mut self, frame_index: usize) {
        self.vert_uniform.update(&self.camera);
        self.frag_uniform.update(&self.camera);

        self.uniform_buffers.upload_data(
            frame_index,
            &self.vert_uniform,
            &self.frag_uniform,
        );
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.lights.destroy(device);
        self.light_descriptor.destroy(device);
        self.uniform_buffers.destroy(device);
        self.materials.destroy(device);
        self.models.destroy(device);
    }
}

fn create_texture_sampler(device: &Device) -> Result<vk::Sampler> {
    let create_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(device.device_properties.limits.max_sampler_anisotropy)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);
    Ok(unsafe {
        device.handle.create_sampler(&create_info, None)?
    })
}

#[derive(Default)]
pub struct SceneData {
    pub meshes: Vec<MeshData>,
    pub materials: Vec<MaterialData>,
}

pub struct ImageData {
    pub data: Vec<u8>,
    pub height: u32,
    pub width: u32,
}

pub struct MeshData {
    pub material: usize,
    pub transform: Mat4,
    pub verts: Vec<u8>,
    pub indices: Vec<u8>,
}

pub struct MaterialData {
    pub base_color: ImageData,
    pub metallic: f32,
    pub roughness: f32,
}
