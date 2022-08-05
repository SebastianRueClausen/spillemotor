use anyhow::Result;
use smallvec::SmallVec;
use ash::vk;

use std::{ops, slice};
use std::rc::Rc;

use crate::core::Device;
use crate::util;

type MemoryRange = ops::Range<vk::DeviceSize>;

/// A wrapper around a raw ptr into a mapped memory block.
pub struct MappedMemory {
    ptr: *mut u8,
    block: MemoryBlock,
}

impl MappedMemory {
    pub fn new(block: &MemoryBlock) -> Result<Self> {
        let ptr = unsafe {
            block.device.handle
                .map_memory(block.handle, 0, block.size, vk::MemoryMapFlags::empty())
                .map(|ptr| ptr as *mut u8)?
        };
        Ok(Self { ptr, block: block.clone() })
    }

    unsafe fn get_range_unchecked(&self, range: MemoryRange) -> &mut [u8] {
        let start = self.ptr.offset(range.start as isize);
        let len = range.end - range.start;
        slice::from_raw_parts_mut(start, len as usize)
    }

    /// Get the mapped data of a buffer.
    ///
    /// # Panic
    ///
    /// This will panic if the underlying memory block of the buffer isn't the same as the
    /// underlying block of `self`.
    pub fn get_buffer_data(&self, buffer: &Buffer) -> &mut [u8] {
        if buffer.block != self.block {
            panic!("block of buffer isn't same as mapped memory");
        }
        // Safety: At this point we know that the buffer uses the same memory block, and the buffer
        // will always be in the range of the memory block.
        unsafe { self.get_range_unchecked(buffer.range.clone()) }
    }
}

/// A block of raw device memory.
pub struct MemoryBlockShared {
    handle: vk::DeviceMemory,

    #[allow(dead_code)]
    properties: vk::MemoryPropertyFlags,

    size: vk::DeviceSize,

    device: Device,
}

impl MemoryBlockShared {
    fn new(
        device: &Device,
        properties: vk::MemoryPropertyFlags,
        info: &vk::MemoryAllocateInfo,
    ) -> Result<Self> {
        let handle = unsafe { device.handle.allocate_memory(info, None)? };
        let size = info.allocation_size;

        Ok(Self {
            device: device.clone(),
            handle,
            properties,
            size,
        })
    }

    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
}

impl Drop for MemoryBlockShared {
    fn drop(&mut self) {
        unsafe { self.device.handle.free_memory(self.handle, None); }
    }
}

#[derive(Clone)]
pub struct MemoryBlock {
    shared: Rc<MemoryBlockShared>, 
}

impl ops::Deref for MemoryBlock {
    type Target = MemoryBlockShared;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl PartialEq for MemoryBlock {
    fn eq(&self, other: &Self) -> bool {
        Rc::<MemoryBlockShared>::as_ptr(&self.shared)
            == Rc::<MemoryBlockShared>::as_ptr(&other.shared)
    }
}

impl Eq for MemoryBlock {}

impl MemoryBlock {
    fn new(
        device: &Device,
        properties: vk::MemoryPropertyFlags,
        info: &vk::MemoryAllocateInfo,
    ) -> Result<Self> {
        Ok(Self {
            shared: Rc::new(MemoryBlockShared::new(device, properties, info)?)
        })
    }
}

#[derive(Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub block: MemoryBlock,
    pub range: MemoryRange,
}

impl Buffer {
    pub fn new(
        device: &Device,
        info: &vk::BufferCreateInfo,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let handle = unsafe { device.handle.create_buffer(info, None)? };
        let req = unsafe { device.handle.get_buffer_memory_requirements(handle) };

        let memory_type = memory_type_index(
            &device.memory_properties,
            memory_flags,
            req.memory_type_bits,
        )
        .ok_or_else(|| anyhow!("no compatible memory type"))?;

        let mut alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(req.size)
            .memory_type_index(memory_type);
        let mut alloc_flags = vk::MemoryAllocateFlagsInfo::builder();

        if info.usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            alloc_flags = alloc_flags
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS_KHR)
                .device_mask(1);
            alloc_info = alloc_info.push_next(&mut alloc_flags);
        }

        let block = MemoryBlock::new(device, memory_flags, &alloc_info)?;
        unsafe { device.handle.bind_buffer_memory(handle, block.handle, 0)?; }

        Ok(Self { handle, range: 0..block.size, block })
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.range.end - self.range.start
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { self.block.device.handle.destroy_buffer(self.handle, None); }
    }
}

pub fn create_buffers(
    device: &Device,
    create_infos: &[vk::BufferCreateInfo],
    memory_flags: vk::MemoryPropertyFlags,
    alignment: vk::DeviceSize,
) -> Result<(Vec<Buffer>, MemoryBlock)> {
    let mut memory_type_bits = u32::MAX;
    let mut current_size = 0;

    let buffers: Result<SmallVec<[_; 12]>> = create_infos
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

            Ok((handle, start..end))
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

    let block = MemoryBlock::new(device, memory_flags, &alloc_info)?;

    let buffers: Vec<_> = buffers
        .into_iter()
        .map(|(handle, range)| Buffer {
            block: block.clone(),
            handle,
            range,
        })
        .collect();

    for buffer in &buffers {
        unsafe {
            device.handle.bind_buffer_memory(buffer.handle, block.handle, buffer.range.start)?;
        }
    }

    Ok((buffers, block))
}

/// A single image and image view which doesn't own it's own memory.
///
/// This doesn't own it's own memory device memory, and has therefore implicitly the lifetime of
/// the owning [`Images`] or the [`MemoryBlock`] where the memory is allocated.
///
/// # Safety
///
/// Don't use the buffer after destroying the owning [`Images`] object or deallocating the
/// [`MemoryBlock`].
#[derive(Clone)]
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub layout: vk::ImageLayout,
    pub format: vk::Format,
    pub extent: vk::Extent3D,

    pub range: MemoryRange,
    pub block: MemoryBlock,
}

impl Image {
    pub fn size(&self) -> vk::DeviceSize {
        self.range.end - self.range.start
    }

    pub fn new(
        device: &Device,
        image_info: vk::ImageCreateInfo,
        mut view_info: vk::ImageViewCreateInfo,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let handle = unsafe { device.handle.create_image(&image_info, None)? };
        let requirements = unsafe { device.handle.get_image_memory_requirements(handle) };

        let Some(memory_type) = memory_type_index(
            &device.memory_properties,
            memory_flags,
            requirements.memory_type_bits,
        ) else {
            return Err(anyhow!("no compatible memory type"));
        };

        let block = {
            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            MemoryBlock::new(device, memory_flags, &alloc_info)?
        };

        unsafe { device.handle.bind_image_memory(handle, block.handle, 0)?; }

        let view = unsafe {
            view_info.image = handle;
            device.handle.create_image_view(&view_info, None)?
        };

        Ok(Image {
            layout: image_info.initial_layout,
            extent: image_info.extent,
            format: image_info.format,
            range: 0..requirements.size,
            handle,
            view,
            block,
        })
    }

    /// Transition the layout of the image to `new`.
    ///
    /// For now it handles two transitions (`format` references the current layout of the image):
    ///
    /// | `format`               | `new`                      |
    /// |------------------------|----------------------------|
    /// | `UNDEFINED`            | `TRANSFER_DST_OPTIMAL`     |
    /// | `TRANSFER_DST_OPTIMAL` | `SHADER_READ_ONLY_OPTIMAL` |
    ///
    /// The transition will fail if the transfer doesn't fit into the tabel.
    pub fn transition_layout(&mut self, device: &Device, new: vk::ImageLayout) -> Result<()> {
        device.transfer_with(|cmd| {
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
                    (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                    )
                }
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => {
                    barrier = barrier
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                    (
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    )
                }
                _ => {
                    todo!()
                }
            };

            self.layout = new;

            unsafe {
                let barriers = [barrier.build()];
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
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.block.device.handle.destroy_image(self.handle, None);
            self.block.device.handle.destroy_image_view(self.view, None);
        }
    }
}

pub fn create_images(
    device: &Device,
    image_infos: &[vk::ImageCreateInfo],
    view_infos: &[vk::ImageViewCreateInfo],
    memory_flags: vk::MemoryPropertyFlags,
) -> Result<(Vec<Image>, MemoryBlock)> {
    assert_eq!(
        image_infos.len(),
        view_infos.len(),
        "`image_infos` and `view_infos` are not same length"
    );

    let mut memory_type_bits = u32::MAX;
    let mut current_size = 0;

    struct TempImage {
        handle: vk::Image,
        layout: vk::ImageLayout,
        format: vk::Format,
        extent: vk::Extent3D,
        range: MemoryRange,
    }

    let images: Result<SmallVec<[_; 8]>> = image_infos
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

            Ok(TempImage {
                handle,
                extent: image_info.extent,
                layout: image_info.initial_layout,
                format: image_info.format,
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
        MemoryBlock::new(device, memory_flags, &alloc_info)?
    };

    let images: Result<Vec<_>> = images
        .iter_mut()
        .zip(view_infos.iter())
        .map(|(image, view_info)| unsafe {
            device.handle.bind_image_memory(image.handle, block.handle, image.range.start)?;  

            let mut info = view_info.clone();
            info.image = image.handle;

            let view = device.handle.create_image_view(&info, None)?;

            Ok(Image {
                handle: image.handle,
                layout: image.layout,
                format: image.format,
                extent: image.extent,
                range: image.range.clone(),
                block: block.clone(),
                view,
            })
        })
        .collect();

    Ok((images?, block.clone()))
}

pub struct TextureSampler {
    pub handle: vk::Sampler,
    device: Device,
}

impl TextureSampler {
    pub fn new(device: &Device) -> Result<Self> {
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
        let handle = unsafe { device.handle.create_sampler(&create_info, None)? };
        Ok(Self { handle, device: device.clone() })
    }
}

impl Drop for TextureSampler {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_sampler(self.handle, None);
        }
    }
}

/// Find a memory type index fitting `props`, `flags` and `memory_type_bits`.
///
/// Returns `None` if there is no memory type fits the three parameters.
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
