use anyhow::Result;
use ash::vk;

use std::path::Path;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    ptr: Option<*mut u8>,
    pub size: u64,
    pub memory_flags: vk::MemoryPropertyFlags,
}

impl Buffer {
    pub fn new(
        device: &ash::Device,
        info: &vk::BufferCreateInfo,
        memory_flags: vk::MemoryPropertyFlags,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Result<Self> {
        unsafe {
            let buffer = device.create_buffer(info, None)?;
            let req = device.get_buffer_memory_requirements(buffer);

            let mem_type = memory_type_index(memory_properties, memory_flags, req.memory_type_bits)
                .ok_or_else(|| anyhow!("no compatible memory type"))?;

            let mut alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(req.size)
                .memory_type_index(mem_type);

            let mut flags = vk::MemoryAllocateFlagsInfo::builder();

            if info
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            {
                flags = flags
                    .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS_KHR)
                    .device_mask(1);
                alloc_info = alloc_info.push_next(&mut flags)
            }

            let memory = device.allocate_memory(&alloc_info, None)?;
            device.bind_buffer_memory(buffer, memory, 0)?;

            Ok(Self {
                buffer,
                memory,
                ptr: None,
                memory_flags,
                size: info.size,
            })
        }
    }

    pub fn is_mapped(&self) -> bool {
        self.ptr.is_some()
    }

    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.ptr
    }

    pub fn mapped_slice<T: Sized>(&self) -> Option<&mut [T]> {
        self.ptr.map(|ptr| unsafe {
            let element_count = self.size as usize / std::mem::size_of::<T>();
            std::slice::from_raw_parts_mut(ptr as *mut T, element_count)
        })
    }

    pub fn map(&mut self, device: &ash::Device) -> Result<*mut u8> {
        if !self
            .memory_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Err(anyhow!("trying to map non-host visible buffer"));
        }

        let ptr = if let Some(ptr) = self.ptr { ptr } else {
            unsafe {
                let flags = vk::MemoryMapFlags::empty();
                device
                    .map_memory(self.memory, 0, self.size as u64, flags)
                    .map(|ptr| ptr as *mut u8)?
            }
        };

        self.ptr = Some(ptr);

        Ok(ptr)
    }

    pub fn unmap(&self, device: &ash::Device) {
        if !self.is_mapped() {
            warn!("trying to unmap buffer which aren't mapped");
        } else {
            unsafe { device.unmap_memory(self.memory); }
        }
    }

    pub fn fill<T: Sized>(&mut self, device: &ash::Device, data: &[T]) -> Result<()> {
        let ptr = self.map(device)? as *mut T;

        if data.len() * std::mem::size_of::<T>() > self.size as usize {
            return Err(anyhow!("`data` is larger than buffer"));
        }

        unsafe {
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }

        Ok(())
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct Vertex {
    position: [f32; 3], 
}

pub struct Mesh {
    pub verts: Vec<Vertex>,
}

impl Mesh {
    pub fn from_obj(path: &Path) -> Result<Self> {
        let (models, _) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
       
        let verts = models
            .iter()
            .flat_map(|model| model.mesh.positions.as_slice().chunks(3))
            .map(|chunk| {
                let vert = Vertex {
                    position: [chunk[0], chunk[1], chunk[2]]
                };
                vert
            })
            .collect();

        Ok(Self { verts })
    }

    pub fn size(&self) -> u64 {
        (self.verts.len() * std::mem::size_of::<Vertex>()) as u64
    }
}

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
