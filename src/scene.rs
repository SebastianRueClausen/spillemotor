use glam::{Vec3, Vec4, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::mem;
use std::ops::Index;

use crate::camera::Camera;
use crate::light::{PointLight, Lights};
use crate::core::{
    Device,
    Swapchain,
    RenderPass,
    RenderTargets,
    Pipeline,
    ShaderModule,
    PipelineRequest,
    DescriptorSet,
    DescriptorBinding,
    BindingKind,
    CameraUniforms,
};
use crate::resource::{self, Buffer, Image, MappedMemory, TextureSampler};

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
    pub images: Vec<Image>, 
    pub sampler: TextureSampler,
    materials: Vec<Material>,
}

impl Index<usize> for Materials {
    type Output = Material;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.materials[idx]
    }
}

pub struct Scene {
    pub lights: Lights,
    pub render_pipeline: Pipeline,
    pub light_descriptor: DescriptorSet,
    pub materials: Materials,
    pub models: Vec<Model>,
    pub buffers: Vec<Buffer>,
}

impl Scene {
    pub fn from_scene_data(
        device: &Device,
        swapchain: &Swapchain,
        render_pass: &RenderPass,
        render_targets: &RenderTargets,
        camera_uniforms: &CameraUniforms,
        camera: &Camera,
        scene_data: &SceneData,
    ) -> Result<Self> {
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

        let lights = Lights::new(
            device,
            &camera_uniforms,
            &camera,
            &swapchain,
            &lights,
        )?;

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

            let (buffers, block) = resource::create_buffers(device, &create_infos, memory_flags, 4)?;
            let mapped = MappedMemory::new(&block)?;

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
                device,
                &create_infos,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                4,
            )?;

            buffers
        };

        device.transfer_with(|recorder| {
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

            let (staging, block) = resource::create_buffers(device, &create_infos, memory_flags, 4)?;
            let mapped = MappedMemory::new(&block)?;
           
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
            let image_infos: Vec<_> = scene_data.materials
                .iter()
                .flat_map(|mat| {
                    fn image_info(
                        image_data: &ImageData,
                        format: vk::Format,
                    ) -> vk::ImageCreateInfo {
                        vk::ImageCreateInfo::builder()
                            .image_type(vk::ImageType::TYPE_2D)
                            .usage(
                                vk::ImageUsageFlags::TRANSFER_DST
                                    | vk::ImageUsageFlags::SAMPLED
                            )
                            .format(format)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .extent(vk::Extent3D {
                                width: image_data.width,
                                height: image_data.height,
                                depth: 1,
                            })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .build()
                    }
                    [
                        image_info(&mat.base_color, vk::Format::R8G8B8A8_SRGB),
                        image_info(&mat.normal, vk::Format::R8G8B8A8_UNORM),
                        image_info(&mat.metallic_roughness, vk::Format::R8G8_SRGB),
                    ]
                })
                .collect();

            let view_infos: Vec<_> = scene_data.materials
                .iter()
                .flat_map(|_| {
                    fn view_info(format: vk::Format) -> vk::ImageViewCreateInfo {
                        let subresource_range = vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build();
                        vk::ImageViewCreateInfo::builder()
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .subresource_range(subresource_range)
                            .format(format)
                            .build()
                    }
                    [
                        view_info(vk::Format::R8G8B8A8_SRGB),
                        view_info(vk::Format::R8G8B8A8_UNORM),
                        view_info(vk::Format::R8G8_SRGB),
                    ]
                })
                .collect();

            let (images, _) = resource::create_images(
                device,
                &image_infos,
                &view_infos,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            images
        };

        device.transfer_with(|recorder| {
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

        let light_descriptor = DescriptorSet::new(device, &[
            DescriptorBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                kind: BindingKind::Buffer([
                    &lights.cluster_info.buffer, 
                    &lights.cluster_info.buffer, 
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
                    lights.light_mask_buffer(0), 
                    lights.light_mask_buffer(1), 
                ]),
            },
        ])?;

        let sampler = TextureSampler::new(device)?;

        let materials: Result<Vec<_>> = scene_data.materials
            .iter()
            .enumerate()
            .map(|(base, _)| {
                let base = base * 3;

                let base_color_index = base;
                let normal_index = base + 1;
                let metallic_roughness_index = base + 2;

                let descriptor = DescriptorSet::new(device, &[
                    DescriptorBinding {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        stage: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Buffer([
                            camera_uniforms.view_uniform(0),
                            camera_uniforms.view_uniform(1),
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Buffer([
                            camera_uniforms.proj_uniform(),
                            camera_uniforms.proj_uniform(),
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Image(&sampler, [
                            &images[base_color_index],
                            &images[base_color_index],
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Image(&sampler, [
                            &images[normal_index],
                            &images[normal_index],
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Image(&sampler, [
                            &images[metallic_roughness_index],
                            &images[metallic_roughness_index],
                        ]),
                    },
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
            // Check that all the materials share the same descriptor set layout.
            assert!({
                let mut iter = materials.iter();
                let first = iter.next();

                iter.fold(first, |acc, mat| {
                    acc.and_then(|stored| {
                        if stored.descriptor.layout == mat.descriptor.layout {
                            Some(stored)
                        } else {
                            None
                        }
                    })
                })
                .is_some()
            });

            let descriptor_layout = materials
                .first()
                .unwrap()
                .descriptor
                .layout;

            let vertex_code = include_bytes_aligned_as!(u32, "../shaders/vert.spv");
            let fragment_code = include_bytes_aligned_as!(u32, "../shaders/frag.spv");

            let vertex_module = ShaderModule::new(&device, "main", vertex_code)?;
            let fragment_module = ShaderModule::new(&device, "main", fragment_code)?;

            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

            let pipeline = Pipeline::new(device, PipelineRequest::Render {
                descriptor_layouts: &[
                    descriptor_layout,
                    light_descriptor.layout,
                ],
                push_constants: &[
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .size(mem::size_of::<ModelTransform>() as u32)
                        .offset(0)
                        .build(),
                ],
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
                swapchain,
                render_pass,
                render_targets,
            })?;

            pipeline
        };

        let materials = Materials { images, sampler, materials };

        Ok(Self {
            lights,
            light_descriptor,
            render_pipeline,
            buffers,
            models,
            materials,
        })
    }

    pub fn vertex_buffer(&mut self) -> &Buffer {
        &self.buffers[0]
    }

    pub fn index_buffer(&mut self) -> &Buffer {
        &self.buffers[1]
    }

    pub fn handle_resize(
        &mut self,
        device: &Device,
        camera: &Camera,
        swapchain: &Swapchain,
    ) -> Result<()> {
        self.lights.handle_resize(device, camera, swapchain)
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
