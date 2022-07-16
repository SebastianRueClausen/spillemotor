use glam::{Vec3, Vec4, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::mem;
use std::ops::Index;
use std::time::Duration;

use crate::util;
use crate::camera::{InputState, Camera};
use crate::light::{PointLight, Lights};
use crate::core::{Device, Swapchain, RenderPass, RenderTargets, Pipeline};
use crate::core::{DescriptorSet, DescriptorBinding, BindingKind, DescriptorLayoutCache};
use crate::core::CameraUniforms;
use crate::resource::{Images, Buffers};

#[repr(C)]
#[derive(Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub texcoord: Vec2,
    pub tangent: Vec4,
}

pub struct ModelTransform {
    #[allow(dead_code)]
    transform: Mat4,

    #[allow(dead_code)]
    inverse_transpose_transform: Mat4,
}

unsafe impl util::Pod for ModelTransform {}

pub struct Model {
    pub index_buffer: usize,
    pub vertex_buffer: usize,  
    pub index_count: u32,
    pub material: usize,

    transform: ModelTransform,
}

impl Model {
    pub fn transform(&self) -> &ModelTransform {
        &self.transform
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
    pub camera: Camera,

    camera_uniforms: CameraUniforms,

    pub render_pipeline: Pipeline,
    pub light_descriptor: DescriptorSet,
    pub materials: Materials,
    pub models: Models,
}

impl Scene {
    pub fn from_scene_data(
        device: &Device,
        layout_cache: &mut DescriptorLayoutCache,
        swapchain: &Swapchain,
        render_pass: &RenderPass,
        render_targets: &RenderTargets,
        scene_data: &SceneData,
    ) -> Result<Self> {
        let camera = Camera::new(swapchain.aspect_ratio());
        let camera_uniforms = CameraUniforms::new(device, &camera, swapchain)?;

        let lights = Lights::new(device, layout_cache, &camera_uniforms, &camera, &swapchain, &[
            PointLight::new(
                Vec3::new(-0.9, 0.25, -0.354),
                Vec3::new(1.0, 0.0, 1.0) * 12.0,
                4.0,
            ),
            PointLight::new(
                Vec3::new(-0.9, 0.25, -0.8),
                Vec3::new(0.0, 1.0, 1.0) * 12.0,
                4.0,
            ),
            PointLight::new(
                Vec3::new(9.33, 0.88, 0.55),
                Vec3::new(0.0, 1.0, 1.0) * 20.0,
                4.0,
            ),
        ])?;

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

        staging.block.map_with(device, |mapped| {
            scene_data.meshes.iter()
                .flat_map(|mesh| [mesh.verts.as_slice(), mesh.indices.as_slice()])
                .zip(staging.buffers.iter())
                .for_each(|(src, dst)| unsafe {
                    mapped.get_range(dst.range.clone()).copy_from_slice(&src);
                });
        })?;

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

        for image in images.images.iter_mut() {
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

        for image in images.images.iter_mut() {
            image.transition_layout(device, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        }

        let models: Vec<_> = scene_data.meshes
            .iter()
            .enumerate()
            .map(|(i, mesh)| {
                let transform = ModelTransform {
                    transform: mesh.transform,
                    inverse_transpose_transform: mesh.transform.inverse().transpose(),
                };

                Model {
                    material: mesh.material,
                    transform,
                    vertex_buffer: i * 2,
                    index_buffer: i * 2 + 1,
                    index_count: (mesh.indices.len() / mem::size_of::<u16>()) as u32,
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
                    lights.light_index_buffer(0), 
                    lights.light_index_buffer(1), 
                ]),
            },
        ])?;

        let sampler = create_texture_sampler(device)?;
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

            Pipeline::new_render(
                device,
                swapchain,
                render_pass,
                render_targets,
                &[descriptor_layout, light_descriptor.layout],
            )?
        };

        Ok(Self {
            camera,
            lights,
            light_descriptor,
            camera_uniforms,
            render_pipeline,
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

    pub fn handle_resize(&mut self, device: &Device, swapchain: &Swapchain) -> Result<()> {
        self.camera.update_proj(swapchain.aspect_ratio());
        self.camera_uniforms.update_proj(&self.camera, swapchain);
        self.lights.handle_resize(device, &self.camera, swapchain)
    }

    pub fn upload_data(&mut self, frame_index: usize) {
        self.camera_uniforms.update_view(frame_index, &self.camera);
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.lights.destroy(device);
        self.light_descriptor.destroy(device);
        self.camera_uniforms.destroy(device);
        self.materials.destroy(device);
        self.models.destroy(device);
        self.render_pipeline.destroy(device);
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
    pub normal: ImageData,
    pub metallic_roughness: ImageData,
}
