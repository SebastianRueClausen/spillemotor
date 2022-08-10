use anyhow::Result;
use smallvec::SmallVec;
use ash::vk;

use std::{ops, slice};
use std::rc::Rc;
use std::cell::Cell;

use crate::core::{Renderer, Device};
use crate::handle::Handle;

type MemoryRange = ops::Range<vk::DeviceSize>;

/// A wrapper around a raw ptr into a mapped memory block.
pub struct MappedMemory {
    ptr: *mut u8,
    block: Handle<MemoryBlock>,
}

impl MappedMemory {
    pub fn new(block: Handle<MemoryBlock>) -> Result<Self> {
        let ptr = unsafe {
            block.device.handle
                .map_memory(block.handle, 0, block.size, vk::MemoryMapFlags::empty())
                .map(|ptr| ptr as *mut u8)?
        };
        Ok(Self { ptr, block })
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
        if *buffer.block != *self.block {
            panic!("block of buffer isn't same as mapped memory");
        }
        // Safety: At this point we know that the buffer uses the same memory block, and the buffer
        // will always be in the range of the memory block.
        unsafe { self.get_range_unchecked(buffer.range.clone()) }
    }
}

/// A block of raw device memory.
#[derive(Clone)]
pub struct MemoryBlock {
    handle: vk::DeviceMemory,

    #[allow(dead_code)]
    properties: vk::MemoryPropertyFlags,

    size: vk::DeviceSize,

    device: Device,
}

impl MemoryBlock {
    fn new(
        device: &Device,
        properties: vk::MemoryPropertyFlags,
        info: &vk::MemoryAllocateInfo,
    ) -> Result<Handle<Self>> {
        let handle = unsafe { device.handle.allocate_memory(info, None)? };
        let size = info.allocation_size;
        Ok(Handle::new(Self { device: device.clone(), handle, properties, size }))
    }

    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        unsafe { self.device.handle.free_memory(self.handle, None); }
    }
}

impl PartialEq for MemoryBlock {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for MemoryBlock {}

#[derive(Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub block: Handle<MemoryBlock>,
    pub range: MemoryRange,
}

impl Buffer {
    pub fn new(
        renderer: &Renderer,
        info: &vk::BufferCreateInfo,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<Rc<Self>> {
        let device = renderer.device.clone();

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

        let block = MemoryBlock::new(&device, memory_flags, &alloc_info)?;
        unsafe { device.handle.bind_buffer_memory(handle, block.handle, 0)?; }

        Ok(Rc::new(Self { handle, range: 0..block.size, block }))
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
    renderer: &Renderer,
    create_infos: &[vk::BufferCreateInfo],
    memory_flags: vk::MemoryPropertyFlags,
    alignment: vk::DeviceSize,
) -> Result<(Vec<Rc<Buffer>>, Handle<MemoryBlock>)> {
    let device = renderer.device.clone();

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
            let alignment = lcm(alignment, requirements.alignment);

            // Round `current_size` up to the next integer which has the alignment of
            // `alignment`.
            let start = align_up_to(current_size, alignment);
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

    let block = MemoryBlock::new(&device, memory_flags, &alloc_info)?;

    let buffers: Vec<_> = buffers
        .into_iter()
        .map(|(handle, range)| Rc::new(Buffer {
            block: block.clone(),
            handle,
            range,
        }))
        .collect();

    for buffer in &buffers {
        unsafe {
            device.handle.bind_buffer_memory(buffer.handle, block.handle, buffer.range.start)?;
        }
    }

    Ok((buffers, block))
}

#[derive(Clone, Copy)]
pub struct ImageReq {
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
}

impl Into<vk::ImageCreateInfo> for ImageReq {
    fn into(self) -> vk::ImageCreateInfo {
        vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .format(self.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .extent(self.extent)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .array_layers(1)
            .build()
    }
}

impl Into<vk::ImageViewCreateInfo> for ImageReq {
    fn into(self) -> vk::ImageViewCreateInfo {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(*subresource_range)
            .format(self.format)
            .build()
    }
}

#[derive(Clone)]
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent3D,
    pub format: vk::Format,

    /// Layout is mutable.
    pub layout: Cell<vk::ImageLayout>,

    pub range: MemoryRange,
    pub block: Handle<MemoryBlock>,
}

impl Image {
    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        self.range.end - self.range.start
    }

    pub fn new(
        renderer: &Renderer,
        memory_flags: vk::MemoryPropertyFlags,
        req: ImageReq,
    ) -> Result<Rc<Self>> {
        Self::from_raw(&renderer.device, memory_flags, &req, &req)
    }

    pub fn from_raw<I, V>(
        device: &Device,
        memory_flags: vk::MemoryPropertyFlags,
        image_info: &I,
        view_info: &V,
    ) -> Result<Rc<Self>>
    where
        I: Into<vk::ImageCreateInfo> + Clone + Copy,
        V: Into<vk::ImageViewCreateInfo> + Clone + Copy,
    {
        let device = device.clone();

        let image_info: vk::ImageCreateInfo = (*image_info).into();

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
            MemoryBlock::new(&device, memory_flags, &alloc_info)?
        };

        unsafe { device.handle.bind_image_memory(handle, block.handle, 0)?; }
        
        let mut view_info: vk::ImageViewCreateInfo = (*view_info).into();
        view_info.image = handle;

        let view = unsafe { device.handle.create_image_view(&view_info, None)? };

        Ok(Rc::new(Image {
            layout: Cell::new(image_info.initial_layout),
            extent: image_info.extent,
            format: image_info.format,
            range: 0..requirements.size,
            handle,
            view,
            block,
        }))
    }

    pub fn layout(&self) -> vk::ImageLayout {
        self.layout.get()
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
    renderer: &Renderer,
    memory_flags: vk::MemoryPropertyFlags,
    reqs: &[ImageReq],
) -> Result<(Vec<Rc<Image>>, Handle<MemoryBlock>)> {
    create_images_raw(&renderer.device, memory_flags, reqs, reqs)
}

pub fn create_images_raw<I, V>(
    device: &Device,
    memory_flags: vk::MemoryPropertyFlags,
    image_infos: &[I],
    view_infos: &[V],
) -> Result<(Vec<Rc<Image>>, Handle<MemoryBlock>)>
where
    I: Into<vk::ImageCreateInfo> + Clone + Copy,
    V: Into<vk::ImageViewCreateInfo> + Clone + Copy,
{
    let device = device.clone();

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
        .map(|info| {
            let image_info: vk::ImageCreateInfo = (*info).into();

            let handle = unsafe { device.handle.create_image(&image_info, None)? };
            let requirements = unsafe { device.handle.get_image_memory_requirements(handle) };

            let start = align_up_to(current_size, requirements.alignment);
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
        MemoryBlock::new(&device, memory_flags, &alloc_info)?
    };

    let images: Result<Vec<_>> = images
        .iter_mut()
        .zip(view_infos.iter())
        .map(|(image, info)| unsafe {
            device.handle.bind_image_memory(image.handle, block.handle, image.range.start)?;  

            let mut info: vk::ImageViewCreateInfo = (*info).into();
            info.image = image.handle;

            let view = device.handle.create_image_view(&info, None)?;

            Ok(Rc::new(Image {
                handle: image.handle,
                layout: Cell::new(image.layout),
                format: image.format,
                extent: image.extent,
                range: image.range.clone(),
                block: block.clone(),
                view,
            }))
        })
        .collect();

    Ok((images?, block))
}

#[derive(Clone)]
pub struct TextureSampler {
    pub handle: vk::Sampler,
    device: Device,
}

impl TextureSampler {
    pub fn new(renderer: &Renderer) -> Result<Handle<Self>> {
        let device = renderer.device.clone(); 
        let create_info = vk::SamplerCreateInfo::builder() .mag_filter(vk::Filter::LINEAR) .min_filter(vk::Filter::LINEAR)
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
        Ok(Handle::new(Self { handle, device }))
    }
}

impl Drop for TextureSampler {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_sampler(self.handle, None); }
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

/// Kernighans algorithm.
#[inline]
fn is_pow2(a: u64) -> bool {
    a != 0 && (a & (a - 1)) == 0
}

/// Calculate least common multiple for two powers of 2.
#[inline]
pub fn lcm(a: u64, b: u64) -> u64 {
    if is_pow2(a) && is_pow2(b) {
        a.max(b)
    } else {
        a * (b / gcd(a, b))
    }
}

/// Calculate greatest common divisor.
#[inline]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    // Optimize if `a` and `b` are both powers of 2.
    if is_pow2(a) && is_pow2(b) {
        a.min(b)
    } else {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }
}

/// Round `a` up to next integer with aligned to `aligment`.
#[inline]
pub fn align_up_to(a: u64, alignment: u64) -> u64 {
    ((a + alignment - 1) / alignment) * alignment
}

#[test]
fn test_lcm() {
    assert_eq!(lcm(5, 12), 60);
    assert_eq!(lcm(4, 64), 64);
    assert_eq!(lcm(2, 5), 10);
    assert_eq!(lcm(3, 4), 12);
    assert_eq!(lcm(2, 4), 4);
}

#[test]
fn test_gcd() {
    assert_eq!(gcd(5, 12), 1);
    assert_eq!(gcd(4, 64), 4);
    assert_eq!(gcd(2, 5), 1);
    assert_eq!(gcd(3, 4), 1);
    assert_eq!(gcd(2, 4), 2);
    assert_eq!(gcd(10, 12), 2);
}
