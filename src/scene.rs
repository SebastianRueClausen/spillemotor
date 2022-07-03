use glam::{Vec3, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::{fs, io, mem};
use std::path::{PathBuf, Path};
use std::ops::Index;

use crate::util;
use crate::core::{Device, Swapchain, RenderPass, RenderTargets, UniformBuffers, Pipeline};
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
    pub materials: Materials,
    pub models: Models,
}

impl Scene {
    pub fn from_scene_data(
        device: &Device,
        swapchain: &Swapchain,
        render_pass: &RenderPass,
        render_targets: &RenderTargets,
        uniform_buffers: &UniformBuffers,
        scene_data: &SceneData,
    ) -> Result<Self> {
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
                mapped.get_range(&dst.range).copy_from_slice(&src);
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

        device.transfer(|command_buffer| {
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
                mapped.get_range(&dst.range).copy_from_slice(&src);
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

        device.transfer(|command_buffer| {
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

        let sampler = create_texture_sampler(device)?;

        let materials: Result<Vec<_>> = scene_data.materials
            .iter()
            .enumerate()
            .map(|(base_color, mat)| {
                let view = images[base_color].view;
                let pipeline = Pipeline::new_render(
                    device,
                    swapchain,
                    render_pass,
                    render_targets,
                    uniform_buffers,
                    view,
                    sampler,            
                )?;
                Ok(Material {
                    base_color,
                    pipeline,
                    params: MaterialParams {
                        metallic: mat.metallic,
                        roughness: mat.roughness,
                    },
                })
            })
            .collect();

        let materials = materials?;

        Ok(Self {
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

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
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

pub struct SceneData {
    meshes: Vec<MeshData>,
    materials: Vec<MaterialData>,
}

struct ImageData {
    data: Vec<u8>,
    height: u32,
    width: u32,
}

struct MeshData {
    material: usize,
    transform: Mat4,
    verts: Vec<u8>,
    indices: Vec<u8>,
}

struct MaterialData {
    base_color: ImageData,
    metallic: f32,
    roughness: f32,
}

impl SceneData {
    pub fn from_gltf(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        let gltf = gltf::Gltf::from_reader(reader)?;

        let buffer_data: Result<Vec<_>> = gltf
            .buffers()
            .map(|buffer| {
                Ok(match buffer.source() {
                    gltf::buffer::Source::Bin => {
                        gltf.blob
                            .as_ref()
                            .map(|blob| blob.clone().into_boxed_slice())
                            .ok_or_else(|| anyhow!("no binary blob in gltf scene"))?
                    }
                    gltf::buffer::Source::Uri(uri) => {
                        let path: PathBuf = [path.parent().unwrap(), Path::new(uri)]
                            .iter()
                            .collect();
                        fs::read(&path)?.into_boxed_slice()
                    }
                })
            })
            .collect();

        let buffer_data = buffer_data?;

        let get_buffer_data = |
            view: &gltf::buffer::View,
            offset: usize,
            size: Option<usize>,
        | -> Vec<u8>{
            let start = view.offset() + offset;
            let end = view.offset() + offset + size.unwrap_or(view.length() - offset);
            buffer_data[view.buffer().index()][start..end].to_vec()
        };

        let load_image = |source: &gltf::image::Source| -> Result<ImageData> {
            match source {
                gltf::image::Source::View { view, mime_type } => {
                    let format = match *mime_type {
                        "image/png" => image::ImageFormat::Png,
                        "image/jpeg" => image::ImageFormat::Jpeg,
                        _ => return Err(anyhow!("image must be either png of jpeg")),
                    };

                    let input = get_buffer_data(view, 0, None);

                    let image = image::load(io::Cursor::new(&input), format)?.into_rgba8();
                    let data = image.as_raw().to_vec();

                    Ok(ImageData {
                        data,
                        width: image.width(),
                        height: image.height(),
                    })
                }
                gltf::image::Source::Uri { uri, .. } => {
                    let uri = Path::new(uri);

                    let path: PathBuf = [path.parent().unwrap(), Path::new(uri)]
                        .iter()
                        .collect();

                    let image = image::open(&path)?.into_rgba8();
                    let data = image.as_raw().to_vec();

                    Ok(ImageData {
                        data,
                        width: image.width(),
                        height: image.height(),
                    })
                }
            }
        };

        let materials: Result<Vec<_>> = gltf
            .materials()
            .map(|mat| {
                let base_color = mat
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .ok_or_else(|| anyhow!("no base color texture on material"))?
                    .texture()
                    .source()
                    .source();
                let metallic = mat
                    .pbr_metallic_roughness()
                    .metallic_factor();
                let roughness = mat
                    .pbr_metallic_roughness()
                    .roughness_factor();
                Ok(MaterialData {
                    base_color: load_image(&base_color)?,
                    metallic,
                    roughness,
                })
            })
            .collect();

        let materials = materials?;

        let meshes: Result<Vec<_>> = gltf.nodes()
            .filter_map(|node| {
                node.mesh().map(|mesh| (mesh, node.transform().matrix()))
            })
            .map(|(mesh, transform)| {
                let Some(primitive) = mesh.primitives().nth(0) else {
                    return Err(anyhow!("mesh {} has no primitives", mesh.index()));
                };

                let Some(material) = primitive.material().index() else {
                    return Err(anyhow!("primitive {} doesn't have a material", primitive.index()));
                };

                let verts = {
                    let positions = primitive.get(&gltf::Semantic::Positions);
                    let normals = primitive.get(&gltf::Semantic::Normals);
                    let texcoords = primitive.get(&gltf::Semantic::TexCoords(0));

                    let (Some(positions), Some(texcoords), Some(normals)) =
                        (positions, texcoords, normals) else
                    {
                        return Err(anyhow!(
                            "prmitive {} doesn't have both positions, texcoords and normals",
                            primitive.index(),
                        ));
                    };

                    use gltf::accessor::{DataType, Dimensions};

                    fn check_format(
                        acc: &gltf::Accessor,
                        dt: DataType,
                        d: Dimensions,
                    ) -> Result<()> {
                        if dt != acc.data_type() {
                            return Err(anyhow!( "accessor {} must be a {dt:?}", acc.index()));
                        }
                        if d != acc.dimensions() {
                            return Err(anyhow!( "accessor {} must be a {d:?}", acc.index()));
                        }
                        Ok(())
                    }
                
                    check_format(&positions, DataType::F32, Dimensions::Vec3)?;
                    check_format(&normals, DataType::F32, Dimensions::Vec3)?;
                    check_format(&texcoords, DataType::F32, Dimensions::Vec2)?;

                    let get_accessor_data = |acc: &gltf::Accessor| -> Result<Vec<u8>> {
                        let Some(view) = acc.view() else {
                            return Err(anyhow!("no view on accessor {}", acc.index()));
                        };
                        Ok(get_buffer_data(&view, acc.offset(), Some(acc.count() * acc.size())))
                    };

                    let positions = get_accessor_data(&positions)?;
                    let normals = get_accessor_data(&normals)?;
                    let texcoords = get_accessor_data(&texcoords)?;

                    assert_eq!(positions.len() / 3, texcoords.len() / 2);
                    assert_eq!(positions.len() / 3, normals.len() / 3);

                    positions
                        .as_slice()
                        .chunks(12)
                        .zip(normals.as_slice().chunks(12))
                        .zip(texcoords.as_slice().chunks(8))
                        .flat_map(|((p, n), c)| [
                            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
                            n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8], n[9], n[10], n[11],
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                        ])
                        .collect()
                };

                let indices = {
                    let Some(indices) = primitive.indices() else {
                        return Err(anyhow!("primtive {} has no indices", primitive.index()));
                    };

                    let data_type = indices.data_type();
                    let dimensions = indices.dimensions();

                    use gltf::accessor::{DataType, Dimensions};
                    let (DataType::U16, Dimensions::Scalar) = (data_type, dimensions) else {
                        return Err(anyhow!(
                            "index attribute, accessor {}, must be scalar `u16`",
                            indices.index(),
                        ));
                    };

                    let Some(view) = indices.view() else {
                        return Err(anyhow!("no view on accessor {}", indices.index()));
                    };

                    let offset = indices.offset();
                    let size = indices.count() * indices.size();

                    get_buffer_data(&view, offset, Some(size))
                };
                Ok(MeshData {
                    verts,
                    indices,
                    material,
                    transform: Mat4::from_cols_array_2d(&transform),
                })
            })
            .collect();
        Ok(Self {
            meshes: meshes?,
            materials,
        })
    }
}

