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
    DescriptorLayoutCache,
    CameraUniforms,
};
use crate::resource::{self, Images, Buffers, BufferView};

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

impl Material {
    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
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
    pub lights: Lights,
    pub render_pipeline: Pipeline,
    pub light_descriptor: DescriptorSet,
    pub materials: Materials,
    pub models: Vec<Model>,
    pub buffers: Buffers,
}

impl Scene {
    pub fn from_scene_data(
        device: &Device,
        layout_cache: &mut DescriptorLayoutCache,
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
            layout_cache,
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
            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        staging.block.map_with(device, |mapped| {
            [scene_data.vertex_data.as_slice(), scene_data.index_data.as_slice()]
                .iter()
                .zip(staging.buffers.iter())
                .for_each(|(src, dst)| unsafe {
                    mapped.get_range(dst.range.clone()).copy_from_slice(&src);
                });
        })?;

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
            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        staging.block.map_with(device, |mapped| {
            scene_data.materials
                .iter()
                .flat_map(|mat| [
                    mat.base_color.data.as_slice(),
                    mat.normal.data.as_slice(),
                    mat.metallic_roughness.data.as_slice(),
                ])
                .zip(staging.iter())
                .for_each(|(src, dst)| unsafe {
                    mapped.get_range(dst.range.clone()).copy_from_slice(&src);
                });
        })?;

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

            Images::new(device, &image_infos, &view_infos, vk::MemoryPropertyFlags::DEVICE_LOCAL)?
        };

        for image in images.iter_mut() {
            image.transition_layout(device, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        }

        device.transfer_with(|command_buffer| {
            for (src, dst) in staging.iter().zip(images.iter()) {
                // I think they should be the same size, but in some instance they aren't for some
                // reason. Vulkan doesn't complain so i guess it's alright.
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

        for image in images.iter_mut() {
            image.transition_layout(device, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        }

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

        let light_descriptor = DescriptorSet::new(device, layout_cache, &[
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

        let sampler = resource::create_texture_sampler(device)?;
        let materials: Result<Vec<_>> = scene_data.materials
            .iter()
            .enumerate()
            .map(|(base, _)| {
                let base = base * 3;

                let base_color_index = base;
                let normal_index = base + 1;
                let metallic_roughness_index = base + 2;

                let descriptor = DescriptorSet::new(device, layout_cache, &[
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
                        kind: BindingKind::Image(sampler, [
                            &images[base_color_index],
                            &images[base_color_index],
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Image(sampler, [
                            &images[normal_index],
                            &images[normal_index],
                        ]),
                    },
                    DescriptorBinding {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        kind: BindingKind::Image(sampler, [
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

            unsafe {
                vertex_module.destroy(device); 
                fragment_module.destroy(device); 
            }

            pipeline
        };

        let materials = Materials {
            images,
            sampler,
            materials,
        };

        Ok(Self {
            lights,
            light_descriptor,
            render_pipeline,
            buffers,
            models,
            materials,
        })
    }

    pub fn vertex_buffer(&mut self) -> &BufferView {
        &self.buffers[0]
    }

    pub fn index_buffer(&mut self) -> &BufferView {
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

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.lights.destroy(device);
        self.light_descriptor.destroy(device);
        self.materials.destroy(device);
        self.buffers.destroy(device);
        self.render_pipeline.destroy(device);
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
