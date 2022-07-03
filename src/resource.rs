use anyhow::Result;
use ash::vk;

use std::ops::{Index, Range};
use std::mem;

use crate::core::Device;
use crate::util;

type MemoryRange = Range<vk::DeviceSize>;

/// A wrapper around a raw ptr into a mapped memory block.
///
/// # Safety
///
/// Don't use this after or freeing the [`MemoryBlock`] this is mapped into.
pub struct MappedMemory {
    ptr: *mut u8,
    size: vk::DeviceSize,
}

impl MappedMemory {
    /// Get the memory as slice of an abitrary type.
    ///
    /// The length of the slice is determined by the amount of objects of `T` which can fit into
    /// the mapped memory.
    ///
    /// If the alignment of `T` isn't the same as the aligment of the mapped memory, it will return
    /// `None`.
    pub unsafe fn as_slice<T: Sized>(&self) -> Option<&mut [T]> {
        let Some(ptr) = self.as_ptr() else {
            return None;
        };
        Some(std::slice::from_raw_parts_mut(ptr, self.size as usize / std::mem::size_of::<T>()))
    }

    /// Get the mapped pointer as an abitrary type.
    ///
    /// Returns `None` if the mapped pointer ins't aligned to the alignment of `T`.
    pub unsafe fn as_ptr<T: Sized>(&self) -> Option<*mut T> {
        if self.ptr.align_offset(mem::align_of::<T>()) != 0 {
            return None;
        }
        Some(self.ptr as *mut T)
    }

    /// Get slice of memory in a `range` relative from the start of the buffer.
    ///
    /// # Panic
    ///
    /// If the range isn't contained in the mapped memory.
    pub unsafe fn get_range(&self, range: &MemoryRange) -> &mut [u8] {
        let start = range.start as usize;
        let end = range.end as usize;

        // `as_slice` will only fail if the type isn't aligned with `ptr`, this will never fail for
        // `u8`.
        self.as_slice::<u8>()
            .unwrap()
            .get_mut(start..end)
            .expect("`range` is outside memory")
    }
}

/// A block of device memory.
pub struct MemoryBlock {
    handle: vk::DeviceMemory,
    #[allow(dead_code)]
    properties: vk::MemoryPropertyFlags,
    size: vk::DeviceSize,
}

impl MemoryBlock {
    /// Allocate a new block of device memory.
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

    /// Bind `buffer` to a slice of the memory owned by `self. The memory range is determined by
    /// `buffer` and the function will fail if that range goes past the end of the memory block.
    pub fn bind_buffer(&self, device: &Device, buffer: &BufferView) -> Result<()> {
        unsafe {
            device.handle.bind_buffer_memory(buffer.handle, self.handle, buffer.range.start)?;
        }

        Ok(())
    }

    /// Map the whole memory block.
    /// May fail if the memory isn't host visble.
    pub fn map(&self, device: &Device) -> Result<MappedMemory> {
        let ptr = unsafe {
            let flags = vk::MemoryMapFlags::empty();
            device.handle
                .map_memory(self.handle, 0, self.size, flags)
                .map(|ptr| ptr as *mut u8)?
        };
        Ok(MappedMemory {
            ptr: ptr as *mut u8,
            size: self.size,
        })
    }

    /// Unmap the the memory of self. It must consume a [`MappedMemory`] object to both insure that
    /// the memory is current mapped, and to insure said mapped memory isn't used after calling
    /// this.
    pub fn unmap(&self, device: &Device, _mapped: MappedMemory) {
        unsafe { device.handle.unmap_memory(self.handle); }
    }

    /// Free the memory and leave `self` in an invalid state. This leaves any [`MappedMemory`]
    /// pointing into this block dangling.
    ///
    /// # Safety
    ///
    /// Don't use `self` and any [`MappedMemory`] pointing intp this block after calling this
    /// function.
    pub unsafe fn free(&self, device: &Device) {
        device.handle.free_memory(self.handle, None);
    }
}

/// A single buffer which doesn't own the memory.
///
/// This means that it implicitly has the lifetime the [`MemoryBlock`] owning the memory, and more
/// common it's parent [`Buffers`] object.
///
/// # Safety
///
/// Don't use the buffer after deallocating it's [`MemoryBlock`].
#[derive(Debug, Clone)]
pub struct BufferView {
    pub handle: vk::Buffer,
    pub range: MemoryRange,
}

impl BufferView {
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

/// An aggregate collection of buffers.
///
/// All the buffers are stored in a single [`MemoryBlock`], meaning that all the buffers share the
/// same memory properties.
///
/// This is useful as you often store many buffers with the same lifetime and memory properties.
/// Having them all bunched together have several advantages. It mostly alliviates the need for
/// an general purpose allocator such as VMA, as the number of allocations you need can become very
/// low.
///
/// It's also very convenient when uploading data to the GPU. You could in theory have a whole
/// scene in a big binary blob, and just upload it all in one go to a collection of buffers.
///
/// Copying staging buffers to device buffers also becomes easy as you just create buffers in
/// device memory with the same layout as a collection of buffers in host memory, and go through
/// each pair of buffers and copy them over.
pub struct Buffers {
    pub block: MemoryBlock,
    pub buffers: Vec<BufferView>,
}

impl Index<usize> for Buffers {
    type Output = BufferView;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.buffers[idx]
    }
}

impl Buffers {
    /// Buffers are often stored in chunks of the same size. This is an easy way to iterate over
    /// said chunks.
    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = &[BufferView]> {
        self.buffers.as_slice().chunks(chunk_size)
    }

    pub fn iter(&self) -> impl Iterator<Item = &BufferView> {
        self.buffers.iter()
    }

    /// Create new collection of buffers allocated in a single memory block.
    ///
    /// The info about each buffer is determined by `create_infos`, and the resulting buffers will
    /// be in the same order.
    ///
    /// All the buffers are guaranteed to be aligned to `alignment`.
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

                Ok(BufferView {
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
pub struct ImageView {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub layout: vk::ImageLayout,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub range: MemoryRange,
}

impl ImageView {
    /// The size of the image in bytes.
    pub fn size(&self) -> vk::DeviceSize {
        self.range.end - self.range.start
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

/// An aggregate collection of images all located in a single memory block.
pub struct Images {
    pub images: Vec<ImageView>,
    pub block: MemoryBlock,
}

impl Index<usize> for Images {
    type Output = ImageView;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.images[idx]
    }
}

impl Images {
    pub fn iter(&self) -> impl Iterator<Item = &ImageView> {
        self.images.iter()
    }

    /// Images are often stored in chunks of the same size. This is an easy way to iterate over
    /// said chunks.
    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = &[ImageView]> {
        self.images.as_slice().chunks(chunk_size)
    }

    /// Allocate and create a collection of images and image infos.
    ///
    /// # Panic
    ///
    /// * If the length of `image_infos` and `view_infos` aren't the same.
    /// * If the format of the same index into `image_infos` and `view_infos` aren't the same.
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

                Ok(ImageView {
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
