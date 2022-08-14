use glam::{Vec3, Vec4, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::mem;
use std::ops::Index;

use crate::light::Lights;
use crate::core::*;
use crate::camera::CameraUniforms;
use crate::resource::{
    self, Buffer, Image, ImageReq, MappedMemory, TextureSampler, ResourcePool, Res,
};

#[repr(C)]
#[derive(Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoord: Vec2,
    pub tangent: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct ModelTransform {
    #[allow(dead_code)]
    transform: Mat4,

    #[allow(dead_code)]
    inverse_transpose_transform: Mat4,
}

pub struct Model {
    pub index_start: u32,
    pub index_count: u32,
    pub vertex_start: u32,
    
    pub material: usize,

    transform: ModelTransform,
}

impl Model {
    pub fn transform(&self) -> &ModelTransform {
        &self.transform
    }
}

pub struct Material {
    pub base_color: usize,
    pub normal: usize,
    pub metallic_roughness: usize,
    pub descriptor: DescriptorSet,
}

pub struct Materials {
    pub images: Vec<Res<Image>>, 
    pub sampler: Res<TextureSampler>,
    materials: Vec<Material>,
}

impl Index<usize> for Materials {
    type Output = Material;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.materials[idx]
    }
}

pub struct Scene {
    pub render_pipeline: GraphicsPipeline,
    pub light_descriptor: DescriptorSet,
    pub materials: Materials,
    pub models: Vec<Model>,
    pub buffers: Vec<Res<Buffer>>,
}

impl Scene {
    pub fn from_scene_data(
        renderer: &Renderer,
        pool: &ResourcePool,
        camera_uniforms: &CameraUniforms,
        lights: &Lights,
        scene_data: &SceneData,
    ) -> Result<Self> {
        let staging = {
            let create_infos = [
                vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .size(scene_data.vertex_data.len() as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .size(scene_data.index_data.len() as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
            ];

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let (buffers, block) = resource::create_buffers(
                &renderer,
                &pool,
                &create_infos,
                memory_flags,
                4,
            )?;

            let mapped = MappedMemory::new(block.clone())?;

            [scene_data.vertex_data.as_slice(), scene_data.index_data.as_slice()]
                .iter()
                .zip(buffers.iter())
                .for_each(|(data, buffer)| {
                    mapped.get_buffer_data(buffer).copy_from_slice(&data);
                });

            buffers
        };

        let buffers = {
            let create_infos = [
                vk::BufferCreateInfo::builder()
                    .usage(
                        vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                    )
                    .size(scene_data.vertex_data.len() as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                vk::BufferCreateInfo::builder()
                    .usage(
                        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST
                    )
                    .size(scene_data.index_data.len() as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
            ];

            let (buffers, _) = resource::create_buffers(
                &renderer,
                &pool,
                &create_infos,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                4,
            )?;

            buffers
        };

        renderer.device.transfer_with(|recorder| {
            for (src, dst) in staging.iter().zip(buffers.iter()) {
                recorder.copy_buffers(src, dst);
            }
        })?;

        let staging = {
            let create_infos: Vec<_> = scene_data.materials
                .iter()
                .flat_map(|mat| [
                    vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .size(mat.base_color.data.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                    vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .size(mat.normal.data.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                    vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .size(mat.metallic_roughness.data.len() as u64)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .build(),
                ])
                .collect();

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let (staging, block) = resource::create_buffers(
                &renderer,
                &pool,
                &create_infos,
                memory_flags,
                4,
            )?;

            let mapped = MappedMemory::new(block.clone())?;
           
            scene_data.materials
                .iter()
                .flat_map(|mat| [
                    mat.base_color.data.as_slice(),
                    mat.normal.data.as_slice(),
                    mat.metallic_roughness.data.as_slice(),
                ])
                .zip(staging.iter())
                .for_each(|(data, buffer)| {
                    mapped.get_buffer_data(buffer).copy_from_slice(&data);
                });

            staging
        };

        let mut images = {
            let image_reqs: Vec<_> = scene_data.materials
                .iter()
                .flat_map(|mat| [
                    ImageReq {
                        format: vk::Format::R8G8B8A8_SRGB,
                        usage: vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                        extent: vk::Extent3D {
                            width: mat.base_color.width,
                            height: mat.base_color.height,
                            depth: 1,
                        },
                    },
                    ImageReq {
                        format: vk::Format::R8G8B8A8_UNORM,
                        usage: vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                        extent: vk::Extent3D {
                            width: mat.normal.width,
                            height: mat.normal.height,
                            depth: 1,
                        },
                    },
                    ImageReq {
                        format: vk::Format::R8G8_SRGB,
                        usage: vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                        extent: vk::Extent3D {
                            width: mat.metallic_roughness.width,
                            height: mat.metallic_roughness.height,
                            depth: 1,
                        },
                    },
                ])
                .collect();

            let (images, _) = resource::create_images(
                &renderer,
                &pool,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                &image_reqs,
            )?;

            images
        };

        renderer.device.transfer_with(|recorder| {
            for image in images.iter_mut() {
                recorder.transition_image_layout(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            }

            for (src, dst) in staging.iter().zip(images.iter()) {
                recorder.copy_buffer_to_image(src, dst);
            }

            for image in images.iter_mut() {
                recorder.transition_image_layout(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        })?;

        let models: Vec<_> = scene_data.meshes
            .iter()
            .map(|mesh| {
                let transform = ModelTransform {
                    transform: mesh.transform,
                    inverse_transpose_transform: mesh.transform.inverse().transpose(),
                };

                Model {
                    material: mesh.material,
                    transform,
                    vertex_start: mesh.vertex_start,
                    index_start: mesh.index_start,
                    index_count: mesh.index_count,
                }
            })
            .collect();

        let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
        ])?);

        let light_descriptor = DescriptorSet::new_per_frame(&renderer, layout, &[
            DescriptorBinding::Buffer([
                lights.cluster_info.buffer.clone(), 
                lights.cluster_info.buffer.clone(), 
            ]),
            DescriptorBinding::Buffer([
                lights.light_buffer().clone(), 
                lights.light_buffer().clone(), 
            ]),
            DescriptorBinding::Buffer([
                lights.light_mask_buffer(0).clone(), 
                lights.light_mask_buffer(1).clone(), 
            ]),
        ])?;

        let sampler = pool.alloc(TextureSampler::new(&renderer)?);

        let descriptor_layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
        ])?);

        let materials: Result<Vec<_>> = scene_data.materials
            .iter()
            .enumerate()
            .map(|(base, _)| {
                let base = base * 3;

                let base_color_index = base;
                let normal_index = base + 1;
                let metallic_roughness_index = base + 2;

                let descriptor = DescriptorSet::new_per_frame(&renderer, descriptor_layout.clone(), &[
                    DescriptorBinding::Buffer([
                        camera_uniforms.view_uniform(0).clone(),
                        camera_uniforms.view_uniform(1).clone(),
                    ]),
                    DescriptorBinding::Buffer([
                        camera_uniforms.proj_uniform().clone(),
                        camera_uniforms.proj_uniform().clone(),
                    ]),
                    DescriptorBinding::Image(sampler.clone(), [
                        images[base_color_index].clone(),
                        images[base_color_index].clone(),
                    ]),
                    DescriptorBinding::Image(sampler.clone(), [
                        images[normal_index].clone(),
                        images[normal_index].clone(),
                    ]),
                    DescriptorBinding::Image(sampler.clone(), [
                        images[metallic_roughness_index].clone(),
                        images[metallic_roughness_index].clone(),
                    ]),
                ])?;

                Ok(Material {
                    base_color: base_color_index,
                    metallic_roughness: metallic_roughness_index,
                    normal: normal_index,
                    descriptor,
                })
            })
            .collect();

        let materials = materials?;

        let render_pipeline = {
            let vertex_code = include_bytes_aligned_as!(u32, "../shaders/vert.spv");
            let fragment_code = include_bytes_aligned_as!(u32, "../shaders/frag.spv");

            let vertex_module = ShaderModule::new(&renderer, "main", vertex_code)?;
            let fragment_module = ShaderModule::new(&renderer, "main", fragment_code)?;

            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .size(mem::size_of::<ModelTransform>() as u32)
                .offset(0)
                .build()];

            let layout = pool.alloc(PipelineLayout::new(&renderer, &push_consts, &[
                descriptor_layout.clone(),
                light_descriptor.layout.clone(),
            ])?);

            let pipeline = GraphicsPipeline::new(&renderer, GraphicsPipelineReq {
                layout,
                vertex_attributes: &[
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32B32_SFLOAT,
                        binding: 0,
                        location: 0,
                        offset: 0,
                    },
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32B32_SFLOAT,
                        binding: 0,
                        location: 1,
                        offset: mem::size_of::<Vec3>() as u32,
                    },
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32_SFLOAT,
                        binding: 0,
                        location: 2,
                        offset: mem::size_of::<[Vec3; 2]>() as u32
                    },
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32B32A32_SFLOAT,
                        binding: 0,
                        location: 3,
                        offset: (mem::size_of::<[Vec3; 2]>() + mem::size_of::<Vec2>())as u32,
                    },
                ],
                vertex_bindings: &[vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: mem::size_of::<Vertex>() as u32,
                    input_rate: vk::VertexInputRate::VERTEX,
                }],
                depth_stencil_info: &depth_stencil_info,
                vertex_shader: &vertex_module,
                fragment_shader: &fragment_module,
            })?;

            pipeline
        };

        let materials = Materials { images, sampler, materials };

        Ok(Self { light_descriptor, render_pipeline, buffers, models, materials })
    }

    pub fn vertex_buffer(&self) -> &Buffer {
        &self.buffers[0]
    }

    pub fn index_buffer(&self) -> &Buffer {
        &self.buffers[1]
    }
}

#[derive(Default)]
pub struct SceneData {
    pub meshes: Vec<MeshData>,
    pub materials: Vec<MaterialData>,

    pub vertex_data: Vec<u8>,
    pub index_data: Vec<u8>,
}

pub struct ImageData {
    pub data: Vec<u8>,
    pub height: u32,
    pub width: u32,
}

pub struct MeshData {
    pub material: usize,
    pub transform: Mat4,

    pub index_start: u32,
    pub index_count: u32,
    pub vertex_start: u32,
}

pub struct MaterialData {
    pub base_color: ImageData,
    pub normal: ImageData,
    pub metallic_roughness: ImageData,
}
