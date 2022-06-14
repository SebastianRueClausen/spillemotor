use anyhow::Result;
use ash::vk;

use std::ops::Range;
use std::path::Path;

use crate::core::Device;

type MemoryRange = Range<vk::DeviceSize>;

pub struct MappedMemory {
    ptr: *mut u8,
    size: vk::DeviceSize,
}

impl MappedMemory {
    pub unsafe fn as_slice<T: Sized>(&self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.as_ptr(), self.size as usize / std::mem::size_of::<T>())
    }

    pub unsafe fn as_ptr<T: Sized>(&self) -> *mut T {
        self.ptr as *mut T
    }

    pub unsafe fn get_range(&self, range: &MemoryRange) -> &mut [u8] {
        let start = range.start as usize;
        let end = range.end as usize;
        &mut self.as_slice::<u8>()[start..end]
    }
}

pub struct MemoryBlock {
    handle: vk::DeviceMemory,
    #[allow(dead_code)]
    properties: vk::MemoryPropertyFlags,
    size: vk::DeviceSize,
}

impl MemoryBlock {
    pub fn allocate(
        device: &Device,
        properties: vk::MemoryPropertyFlags,
        info: &vk::MemoryAllocateInfo,
    ) -> Result<Self> {
        let handle = unsafe { device.handle.allocate_memory(info, None)? };
        let size = info.allocation_size;

        Ok(Self {
            handle,
            properties,
            size,
        })
    }

    pub fn bind_buffer(&self, device: &Device, buf: &Buffer) -> Result<()> {
        unsafe {
            device.handle.bind_buffer_memory(buf.handle, self.handle, buf.range.start)?;
        }
        Ok(())
    }

    pub fn map(&self, device: &Device) -> Result<MappedMemory> {
        let ptr = unsafe {
            let flags = vk::MemoryMapFlags::empty();
            device
                .handle
                .map_memory(self.handle, 0, self.size, flags)
                .map(|ptr| ptr as *mut u8)?
        };
        Ok(MappedMemory {
            ptr: ptr as *mut u8,
            size: self.size,
        })
    }

    pub fn unmap(&self, device: &Device) {
        unsafe {
            device.handle.unmap_memory(self.handle);
        }
    }

    /// Free the memory and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn free(&self, device: &Device) {
        device.handle.free_memory(self.handle, None);
    }
}

#[derive(Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    range: MemoryRange,
}

impl Buffer {
    pub fn size(&self) -> vk::DeviceSize {
        self.range.end - self.range.start
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        device.handle.destroy_buffer(self.handle, None);
    }
}

pub struct Buffers {
    pub block: MemoryBlock,
    pub buffers: Vec<Buffer>,
}

impl Buffers {
    pub fn new(
        device: &Device,
        create_infos: &[vk::BufferCreateInfo],
        memory_flags: vk::MemoryPropertyFlags,
        alignment: vk::DeviceSize,
    ) -> Result<Buffers> {
        println!("alignment: {}", alignment);

        let mut memory_type_bits = u32::MAX;
        let mut current_size = 0;

        let buffers: Result<Vec<_>> = create_infos
            .iter()
            .map(|info| unsafe {
                let handle = device.handle.create_buffer(info, None)?;
                let requirements = device.handle.get_buffer_memory_requirements(handle);

                memory_type_bits &= requirements.memory_type_bits;

                // Find an aligment that fits both that of `alignemnt` and the alignment required
                // by the buffer.
                let alignment = lcm_pow2(alignment, requirements.alignment);

                // Round `current_size` up to the next integer which has the alignment of
                // `alignment`.
                let start = align_up_to(current_size, alignment);
                let end = start + requirements.size;

                current_size = end;

                Ok(Buffer {
                    handle,
                    range: start..end,
                })
            })
            .collect();

        let buffers = buffers?;

        let memory_type =
            memory_type_index(&device.memory_properties, memory_flags, memory_type_bits)
                .ok_or_else(|| anyhow!("no compatible memory type"))?;

        let mut alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(current_size)
            .memory_type_index(memory_type);

        let mut flags = vk::MemoryAllocateFlagsInfo::builder();

        if create_infos.iter().any(|info| {
            info.usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        }) {
            flags = flags
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS_KHR)
                .device_mask(1);

            alloc_info = alloc_info.push_next(&mut flags)
        }

        let block = MemoryBlock::allocate(device, memory_flags, &alloc_info)?;

        for buffer in &buffers {
            block.bind_buffer(device, &buffer)?;
        }

        Ok(Self { buffers, block })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        for buffer in &self.buffers {
            buffer.destroy(device);
        }
        self.block.free(device);
    }
}

#[derive(Clone)]
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub range: MemoryRange,
}

impl Image {
    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        device.handle.destroy_image(self.handle, None);
        device.handle.destroy_image_view(self.view, None);
    }
}

pub struct Images {
    pub images: Vec<Image>,
    pub block: MemoryBlock,
}

impl Images {
    pub fn new(
        device: &Device,
        image_infos: &[vk::ImageCreateInfo],
        view_infos: &[vk::ImageViewCreateInfo],
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        assert_eq!(
            image_infos.len(),
            view_infos.len(),
            "`image_infos` and `view_infos` are not same length"
        );

        let mut memory_type_bits = u32::MAX;
        let mut current_size = 0;

        let images: Result<Vec<_>> = image_infos
            .iter()
            .zip(view_infos.iter())
            .map(|(image_info, view_info)| {
                assert_eq!(
                    view_info.format, image_info.format,
                    "image format and view format not the same"
                );

                let handle = unsafe { device.handle.create_image(image_info, None)? };
                let requirements = unsafe { device.handle.get_image_memory_requirements(handle) };

                let start = align_up_to(current_size, requirements.alignment);
                let end = start + requirements.size;

                memory_type_bits &= requirements.memory_type_bits;
                current_size = end;

                Ok(Image {
                    handle,
                    view: vk::ImageView::null(),
                    range: start..end,
                })
            })
            .collect();

        let mut images = images?;

        let memory_type =
            memory_type_index(&device.memory_properties, memory_flags, memory_type_bits)
                .ok_or_else(|| anyhow!("no compatible memory type"))?;

        let block = {
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(current_size)
                .memory_type_index(memory_type);
            MemoryBlock::allocate(device, memory_flags, &alloc_info)?
        };

        for (image, view_info) in images.iter_mut().zip(view_infos.iter()) {
            image.view = unsafe {
                device.handle.bind_image_memory(image.handle, block.handle, image.range.start)?;

                let mut info = view_info.clone();
                info.image = image.handle;

                device.handle.create_image_view(&info, None)?
            };
        }

        Ok(Self { images, block })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    #[allow(dead_code)]
    pub unsafe fn destroy(&self, device: &Device) {
        self.images.iter().for_each(|image| image.destroy(device));
        self.block.free(device);
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct Vertex {
    position: [f32; 3],
}

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    block: MemoryBlock,
}

impl Mesh {
    pub fn from_obj(device: &Device, path: &Path) -> Result<Self> {
        let mesh = MeshData::from_obj(path)?;

        let stagings = {
            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let create_infos = [
                vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .size(mesh.vertex_size as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .size(mesh.index_size as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
            ];

            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        unsafe {
            let mapped = stagings.block.map(device)?;
            mapped
                .get_range(&stagings.buffers[0].range)
                .copy_from_slice(&mesh.data[0..mesh.vertex_size]);
            mapped
                .get_range(&stagings.buffers[1].range)
                .copy_from_slice(&mesh.data[mesh.vertex_size..]);
            stagings.block.unmap(device);
        }

        let buffers = {
            let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

            let create_infos = [
                vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                    .size(mesh.vertex_size as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                vk::BufferCreateInfo::builder()
                    .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                    .size(mesh.index_size as u64)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
            ];

            Buffers::new(device, &create_infos, memory_flags, 4)?
        };

        device.transfer(|command_buffer| {
            for (src, dst) in stagings.buffers.iter().zip(buffers.buffers.iter()) {
                assert_eq!(src.size(), dst.size());

                let regions = [vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(src.size())
                    .build()];

                unsafe {
                    device
                        .handle
                        .cmd_copy_buffer(command_buffer, src.handle, dst.handle, &regions);
                }
            }
        })?;

        unsafe {
            stagings.destroy(device);
        }

        let index_count = (mesh.index_size / std::mem::size_of::<u32>()) as u32;

        let vertex_buffer = buffers.buffers[0].clone();
        let index_buffer = buffers.buffers[1].clone();

        Ok(Self {
            index_count,
            block: buffers.block,
            vertex_buffer,
            index_buffer,
        })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    pub unsafe fn destroy(&self, device: &Device) {
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
        self.block.free(device);
    }
}

struct MeshData {
    data: Vec<u8>,
    vertex_size: usize,
    index_size: usize,
}

impl MeshData {
    fn from_obj(path: &Path) -> Result<Self> {
        let load_options = tobj::LoadOptions {
            single_index: true,
            ignore_lines: true,
            ignore_points: true,
            triangulate: false,
        };

        let (models, _) = tobj::load_obj(path, &load_options)?;

        let mut verts: Vec<_> = models
            .iter()
            .take(1)
            .flat_map(|model| model.mesh.positions.as_slice())
            .flat_map(|pos| pos.to_le_bytes())
            .collect();
        let mut indices: Vec<_> = models
            .iter()
            .take(1)
            .flat_map(|model| model.mesh.indices.as_slice())
            .flat_map(|index| index.to_le_bytes())
            .collect();

        let vertex_size = verts.len();
        let index_size = indices.len();

        verts.append(&mut indices);

        Ok(Self {
            data: verts,
            vertex_size,
            index_size,
        })
    }
}

#[inline]
fn memory_type_index(
    props: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
    memory_type_bits: u32,
) -> Option<u32> {
    props.memory_types[0..(props.memory_type_count as usize)]
        .iter()
        .enumerate()
        .position(|(i, memory_type)| {
            let i = i as u32;
            memory_type_bits & (1 << i) != 0 && (memory_type.property_flags & flags) == flags
        })
        .map(|index| index as u32)
}

/// Calculate greatest common multiple for two powers of 2. Used to calculate alignment, which is
/// all powers of 2.
#[inline]
fn gcm_pow2(mut a: u64, mut b: u64) -> u64 {
    debug_assert_eq!(a % 2, 0);
    debug_assert_eq!(b % 2, 0);

    while b != 0 {
        let t = b;
        b = a & (b - 1);
        a = t;
    }

    a
}

#[inline]
fn lcm_pow2(a: u64, b: u64) -> u64 {
    a * (b / gcm_pow2(a, b))
}

#[inline]
fn align_up_to(a: u64, alignment: u64) -> u64 {
    ((a + alignment - 1) / alignment) * alignment
}
