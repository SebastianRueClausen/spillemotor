use anyhow::Result;
use ash::vk;

use std::ops::{Index, Range};

use crate::core::Device;
use crate::util;

type MemoryRange = Range<vk::DeviceSize>;

pub struct MappedMemory {
    ptr: *mut u8,
    size: vk::DeviceSize,
}

impl MappedMemory {
    pub unsafe fn as_slice<T: Sized>(&self) -> &mut [T] {
        std::slice::from_raw_parts_mut(
            self.as_ptr(),
            self.size as usize / std::mem::size_of::<T>(),
        )
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

#[derive(Debug, Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub range: MemoryRange,
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

impl Index<usize> for Buffers {
    type Output = Buffer;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.buffers[idx]
    }
}

impl Buffers {
    pub fn new(
        device: &Device,
        create_infos: &[vk::BufferCreateInfo],
        memory_flags: vk::MemoryPropertyFlags,
        alignment: vk::DeviceSize,
    ) -> Result<Buffers> {
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
                let alignment = util::lcm(alignment, requirements.alignment);

                // Round `current_size` up to the next integer which has the alignment of
                // `alignment`.
                let start = util::align_up_to(current_size, alignment);
                let end = start + info.size;

                current_size = start + requirements.size;

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

        if create_infos.iter().any(|info|
            info.usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        ) {
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
    pub layout: vk::ImageLayout,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub range: MemoryRange,
}

impl Image {
    pub fn size(&self) -> vk::DeviceSize {
        self.range.end - self.range.start
    }

    pub fn transition_layout(&mut self, device: &Device, new: vk::ImageLayout) -> Result<()> {
        assert_ne!(new, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        device.transfer(|cmd| {
            let subresource = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();
            let mut barrier = vk::ImageMemoryBarrier::builder()
                .image(self.handle)
                .old_layout(self.layout)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(subresource);
       
            let (src_stage, dst_stage) = match (self.layout, new) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
                    barrier = barrier
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                    (vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER)
                }
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
                    barrier = barrier
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                    (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER)
                }
                _ => {
                    todo!()  
                },
            };

            self.layout = new;
   
            let barriers = [barrier.build()];

            unsafe {
                device.handle.cmd_pipeline_barrier(
                    cmd,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }
        })?;

        Ok(())
    }

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

impl Index<usize> for Images {
    type Output = Image;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.images[idx]
    }
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

                let start = util::align_up_to(current_size, requirements.alignment);
                let end = start + requirements.size;

                memory_type_bits &= requirements.memory_type_bits;
                current_size = end;

                Ok(Image {
                    handle,
                    extent: image_info.extent,
                    layout: image_info.initial_layout,
                    format: image_info.format,
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
