use anyhow::Result;
use ash::extensions::{ext, khr};
use ash::vk;
use glam::Mat4;

use std::ffi::{self, CStr, CString};
use std::path::Path;

use crate::camera::Camera;
use crate::resource::{Buffers, MappedMemory, Mesh, Vertex};

pub struct Renderer {
    device: Device,
    swapchain: Swapchain,
    render_pass: RenderPass,
    frame_queue: FrameQueue,
    uniform_buffers: UniformBuffers,
    uniform_data: UniformData,
    render_pipeline: Pipeline,
    mesh: Mesh,
    pub camera: Camera,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let device = Device::new(window)?;

        let window_extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        let swapchain = Swapchain::new(&device, window_extent)?;
        let render_pass = RenderPass::new(&device, &swapchain)?;
        let frame_queue = FrameQueue::new(&device)?;

        let uniform_buffers = UniformBuffers::new(&device)?;
        let uniform_data = UniformData::default();

        let render_pipeline =
            Pipeline::new_render(&device, &swapchain, &render_pass, &uniform_buffers)?;

        let mesh = Mesh::from_obj(&device, Path::new("models/lowcat/cat.obj"))?;

        let aspect_ratio = window_extent.width as f32 / window_extent.height as f32;
        let camera = Camera::new(aspect_ratio);

        Ok(Self {
            device,
            swapchain,
            render_pass,
            frame_queue,
            uniform_buffers,
            uniform_data,
            render_pipeline,
            mesh,
            camera,
        })
    }

    pub fn draw(&mut self) -> Result<bool> {
        self.frame_queue.next_frame();

        let frame = self.frame_queue.current_frame();

        // Wait for the frame to be ready to draw to.
        unsafe {
            self.device
                .handle
                .wait_for_fences(&[frame.ready_to_draw], true, u64::MAX)?;
        }

        // Reset command for the current frame.
        unsafe {
            let reset_flags = vk::CommandBufferResetFlags::empty();
            self.device.handle.reset_command_buffer(frame.command_buffer, reset_flags)?;
        }

        let next_image = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                frame.presented,
                vk::Fence::null(),
            )
        };

        match next_image {
            // Getting the next image succeded and the swapchain is optimal.
            Ok((image_index, false)) => {
                // Reset the frame now that we have acquired the image successfully.
                unsafe {
                    self.device.handle.reset_fences(&[frame.ready_to_draw])?;
                }

                // Upload uniform buffer data.
                self.uniform_data.update(&self.camera);
                self.uniform_buffers.upload_data(
                    self.frame_queue.frame_index as u64,
                    &self.uniform_data,
                );

                // Record the command buffer.
                unsafe {
                    let begin_info = vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    self.device.handle.begin_command_buffer(frame.command_buffer, &begin_info)?;

                    let clear_values = [vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.02, 0.02, 0.02, 1.0],
                        },
                    }];

                    let framebuffer = self.render_pass.framebuffers[image_index as usize];

                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(self.render_pass.handle)
                        .framebuffer(framebuffer)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: self.swapchain.extent,
                        })
                        .clear_values(&clear_values);

                    self.device.handle.cmd_begin_render_pass(
                        frame.command_buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    self.device.handle.cmd_bind_pipeline(
                        frame.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.render_pipeline.handle,
                    );

                    let descriptor =
                        self.render_pipeline.descriptor_sets[self.frame_queue.frame_index as usize];

                    self.device.handle.cmd_bind_descriptor_sets(
                        frame.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.render_pipeline.layout,
                        0,
                        &[descriptor],
                        &[],
                    );

                    self.device.handle.cmd_bind_vertex_buffers(
                        frame.command_buffer,
                        0,
                        &[self.mesh.vertex_buffer.handle],
                        &[0]
                    );
                    
                    self.device.handle.cmd_bind_index_buffer(
                        frame.command_buffer,
                        self.mesh.index_buffer.handle,
                        0,
                        vk::IndexType::UINT32,
                    );

                    self.device.handle.cmd_draw_indexed(
                        frame.command_buffer,
                        self.mesh.index_count,
                        1,
                        0,
                        0,
                        0,
                    );

                    self.device.handle.cmd_end_render_pass(frame.command_buffer);
                    self.device.handle.end_command_buffer(frame.command_buffer)?;
                }

                // Submit command buffer to be rendered. Wait for semaphore `sync.presented` first and
                // signals `sync.rendered´ and `sync.ready_to_draw` when all commands have been
                // executed.
                unsafe {
                    let wait = [frame.presented];
                    let command_buffer = [frame.command_buffer];
                    let signal = [frame.rendered];
                    let stage = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

                    let submit_info = [vk::SubmitInfo::builder()
                        .wait_dst_stage_mask(&stage)
                        .wait_semaphores(&wait)
                        .command_buffers(&command_buffer)
                        .signal_semaphores(&signal)
                        .build()];

                    self.device.handle.queue_submit(
                        self.device.graphics_queue.handle,
                        &submit_info,
                        frame.ready_to_draw,
                    )?;
                }

                // Wait for the frame to be rendered before presenting it to the surface.
                unsafe {
                    let wait = [frame.rendered];
                    let swapchains = [self.swapchain.handle];
                    let indices = [image_index];

                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&wait)
                        .swapchains(&swapchains)
                        .image_indices(&indices);

                    self.swapchain
                        .loader
                        .queue_present(self.device.graphics_queue.handle, &present_info)
                        .unwrap();

                    Ok(true)
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Ok((_, true)) => Ok(false),
            Err(result) => Err(result.into()),
        }
    }

    pub fn resize(&mut self, window: &winit::window::Window) -> Result<()> {
        unsafe {
            self.device.handle.device_wait_idle().expect("failed waiting for idle device");
            self.render_pass.destroy(&self.device);
        }

        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        self.swapchain.recreate(&self.device, extent)?;
        self.render_pass = RenderPass::new(&self.device, &self.swapchain)?;

        let aspect_ratio = extent.width as f32 / extent.height as f32;
        self.camera.update_perspective(aspect_ratio);

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .device_wait_idle()
                .expect("failed waiting for idle device");

            self.render_pipeline.destroy(&self.device);
            self.uniform_buffers.destroy(&self.device);
            self.frame_queue.destroy(&self.device);
            self.render_pass.destroy(&self.device);
            self.swapchain.destroy(&self.device);
            self.mesh.destroy(&self.device);
            self.device.destroy();
        }
    }
}

/// The device and data connected to the device used for rendering. This struct data is static
/// after creation and doesn't depend on external factors such as display size.
pub struct Device {
    /// Name of `physical`, for instance "GTX 770".
    name: String,

    pub handle: ash::Device,

    entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,

    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_properties: vk::PhysicalDeviceProperties,

    surface: Surface,

    messenger: DebugMessenger,

    /// Queue used to submit render commands.
    graphics_queue: Queue,

    /// Queue used to submut transfer commands.
    transfer_queue: Queue,

    /// Graphics pool fore recording render commands.
    graphics_pool: vk::CommandPool,

    /// Command pool for recording transfer commands. Not used for any rendering.
    transfer_pool: vk::CommandPool,
}

impl Device {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

        let mut debug_info = {
            use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
            use vk::DebugUtilsMessageTypeFlagsEXT as Type;

            vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    Severity::ERROR | Severity::WARNING | Severity::INFO | Severity::VERBOSE,
                )
                .message_type(Type::GENERAL | Type::PERFORMANCE | Type::VALIDATION)
                .pfn_user_callback(Some(debug_callback))
        };

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
        let layer_names = [validation_layer.as_ptr()];

        let version = vk::make_api_version(0, 1, 0, 0);

        let instance = unsafe {
            let engine_name = CString::new("tobak").unwrap();
            let app_name = CString::new("tobak").unwrap();

            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 0, 0, 1))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 0, 0, 1))
                .api_version(version);

            let ext_names = [
                ext::DebugUtils::name().as_ptr(),
                khr::Surface::name().as_ptr(),
                khr::WaylandSurface::name().as_ptr(),
            ];

            let info = vk::InstanceCreateInfo::builder()
                .push_next(&mut debug_info)
                .application_info(&app_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&ext_names);

            entry.create_instance(&info, None)?
        };

        let surface = Surface::new(&instance, &entry, &window)?;

        let messenger = DebugMessenger::new(&entry, &instance, &debug_info)?;

        let physical = unsafe {
            instance
                .enumerate_physical_devices()?
                .into_iter()
                .max_by_key(|dev| {
                    let properties = instance.get_physical_device_properties(*dev);

                    let mut score = 0;

                    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        score += 1000;
                    }

                    score += properties.limits.max_image_dimension2_d;

                    score
                })
                .ok_or_else(|| anyhow!("no physical devices presented"))?
        };

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let device_properties = unsafe { instance.get_physical_device_properties(physical) };

        let name = unsafe {
            CStr::from_ptr(device_properties.device_name.as_ptr())
                .to_str()
                .unwrap_or("invalid")
                .to_string()
        };

        trace!("using device: {name}");

        let queue_props = unsafe { instance.get_physical_device_queue_family_properties(physical) };

        let graphics_index = queue_props
            .iter()
            .enumerate()
            .position(|(i, p)| {
                p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && unsafe {
                        surface
                            .loader
                            .get_physical_device_surface_support(physical, i as u32, surface.handle)
                            .unwrap_or(false)
                    }
            })
            .map(|index| index as u32)
            .ok_or_else(|| anyhow!("device has no graphics queue"))?;

        let transfer_index = queue_props
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|index| index as u32)
            .unwrap_or(graphics_index);

        let priorities = [1.0_f32];

        let queue_infos = [
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_index)
                .queue_priorities(&priorities)
                .build(),
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(transfer_index)
                .queue_priorities(&priorities)
                .build(),
        ];

        let extensions = [khr::Swapchain::name().as_ptr()];

        // Only create queues for both transfer and graphics if they aren't the same queue.
        let device_info = if transfer_index != graphics_index {
            vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&layer_names)
        } else {
            vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos[1..])
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&layer_names)
        };

        let handle = unsafe { instance.create_device(physical, &device_info, None)? };

        let graphics_queue = Queue::new(&handle, graphics_index);
        let transfer_queue = Queue::new(&handle, transfer_index);

        let graphics_pool = unsafe {
            let info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(graphics_queue.family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            handle.create_command_pool(&info, None)?
        };

        let transfer_pool = unsafe {
            let info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(transfer_queue.family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            handle.create_command_pool(&info, None)?
        };

        Ok(Self {
            name,

            entry,
            instance,
            physical,
            handle,

            memory_properties,
            device_properties,

            surface,
            messenger,

            graphics_queue,
            transfer_queue,

            transfer_pool,
            graphics_pool,
        })
    }

    pub fn transfer(&self, mut func: impl FnMut(vk::CommandBuffer)) -> Result<()> {
        let buffers = unsafe {
            let info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.transfer_pool)
                .command_buffer_count(1);
            self.handle.allocate_command_buffers(&info)?
        };

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.handle.begin_command_buffer(buffers[0], &begin_info)?;
        }

        func(buffers[0]);

        unsafe {
            self.handle.end_command_buffer(buffers[0])?;

            let submit_info = [vk::SubmitInfo::builder().command_buffers(&buffers).build()];

            self.handle.queue_submit(
                self.transfer_queue.handle,
                &submit_info,
                vk::Fence::null(),
            )?;

            self.handle.queue_wait_idle(self.transfer_queue.handle)?;
            self.handle.free_command_buffers(self.transfer_pool, &buffers);
        }

        Ok(())
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self) {
        self.handle.destroy_command_pool(self.graphics_pool, None);
        self.handle.destroy_command_pool(self.transfer_pool, None);
        self.messenger.destroy();
        self.handle.destroy_device(None);
        self.surface.destroy();
        self.instance.destroy_instance(None);
    }
}

pub struct DebugMessenger {
    loader: ext::DebugUtils,
    handle: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        info: &vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> Result<Self> {
        let loader = ext::DebugUtils::new(&entry, &instance);
        let handle = unsafe { loader.create_debug_utils_messenger(&info, None)? };

        Ok(Self { loader, handle })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self) {
        self.loader.destroy_debug_utils_messenger(self.handle, None);
    }
}

pub struct RenderPass {
    handle: vk::RenderPass,
    /// Should always match the swapchain image count.
    framebuffers: Vec<vk::Framebuffer>,
}

impl RenderPass {
    pub fn new(device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let attachments = [vk::AttachmentDescription::builder()
            .format(swapchain.surface_format.format)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .samples(vk::SampleCountFlags::TYPE_1)
            .build()];

        let color_attachment_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let subpass_dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_subpass(0)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build()];

        let handle = unsafe {
            let info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&subpass_dependencies);

            device.handle.create_render_pass(&info, None)?
        };

        let framebuffers: Result<Vec<_>> = swapchain
            .images
            .iter()
            .map(|image| {
                let views = [image.view];
                let info = vk::FramebufferCreateInfo::builder()
                    .render_pass(handle)
                    .attachments(&views)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                Ok(unsafe { device.handle.create_framebuffer(&info, None)? })
            })
            .collect();

        Ok(RenderPass {
            framebuffers: framebuffers?,
            handle,
        })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self, device: &Device) {
        device.handle.destroy_render_pass(self.handle, None);
        for buffer in &self.framebuffers {
            device.handle.destroy_framebuffer(*buffer, None);
        }
    }
}

/// A vulkan queue and it's family index.
struct Queue {
    handle: vk::Queue,
    family_index: u32,
}

impl Queue {
    fn new(device: &ash::Device, family_index: u32) -> Self {
        Self {
            handle: unsafe { device.get_device_queue(family_index, 0) },
            family_index,
        }
    }
}

pub struct Frame {
    /// This get's signaled when the frame has been presented and is then available to draw to
    /// again.
    presented: vk::Semaphore,
    /// This get's signaled when the GPU is done drawing to the frame and the frame is then ready
    /// to be presented.
    rendered: vk::Semaphore,
    /// This is essentially the same as `presented`, but used to sync with the CPU.
    ready_to_draw: vk::Fence,
    /// Command buffer to for drawing and transfers. This has to be re-recorded before each frame,
    /// since the amount of frames (most likely) doesn't match the number if swapchain images.
    command_buffer: vk::CommandBuffer,
}

impl Frame {
    fn new(device: &Device, command_buffer: vk::CommandBuffer) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let presented = unsafe { device.handle.create_semaphore(&semaphore_info, None)? };
        let rendered = unsafe { device.handle.create_semaphore(&semaphore_info, None)? };

        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let ready_to_draw = unsafe { device.handle.create_fence(&fence_info, None)? };

        Ok(Self {
            presented,
            rendered,
            ready_to_draw,
            command_buffer,
        })
    }
}

pub struct FrameQueue {
    /// All the frames in a queue.
    frames: Vec<Frame>,
    /// The index of the frame currently being rendered or presented. It changes just before
    /// rendering of the next image begins.
    frame_index: usize,
}

impl FrameQueue {
    /// Have two frames in flight at a time. You could easily have more, but you risk having the CPU
    /// run multiple frames ahead, which causes latency.
    const FRAME_COUNT: usize = 2;

    pub fn new(device: &Device) -> Result<Self> {
        let command_buffers = unsafe {
            let command_buffer_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(device.graphics_pool)
                .command_buffer_count(Self::FRAME_COUNT as u32);

            device
                .handle
                .allocate_command_buffers(&command_buffer_info)?
        };

        let frames: Result<Vec<_>> = command_buffers
            .into_iter()
            .map(|buffer| Frame::new(device, buffer))
            .collect();

        Ok(Self {
            frames: frames?,
            frame_index: 0,
        })
    }

    pub fn next_frame(&mut self) {
        self.frame_index = (self.frame_index + 1) % Self::FRAME_COUNT;
    }

    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.frame_index]
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self, device: &Device) {
        let command_buffers: Vec<_> = self
            .frames
            .iter()
            .map(|frame| {
                device.handle.destroy_semaphore(frame.rendered, None);
                device.handle.destroy_semaphore(frame.presented, None);
                device.handle.destroy_fence(frame.ready_to_draw, None);

                frame.command_buffer
            })
            .collect();

        device
            .handle
            .free_command_buffers(device.graphics_pool, &command_buffers);
    }
}

/// TODO: Make this handle more display servers besides Wayland.
struct Surface {
    _wayland: khr::WaylandSurface,
    handle: vk::SurfaceKHR,
    loader: khr::Surface,
}

impl Surface {
    fn new(
        instance: &ash::Instance,
        entry: &ash::Entry,
        window: &winit::window::Window,
    ) -> Result<Self> {
        use winit::platform::unix::WindowExtUnix;

        let wayland_display = window
            .wayland_display()
            .ok_or_else(|| anyhow!("not able to fetch wayland display"))?;

        let wayland_surface = window
            .wayland_surface()
            .ok_or_else(|| anyhow!("not able to fetch wayland window"))?;

        let create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
            .display(wayland_display)
            .surface(wayland_surface);

        let wayland = khr::WaylandSurface::new(&entry, &instance);
        let loader = khr::Surface::new(&entry, &instance);

        let handle = unsafe { wayland.create_wayland_surface(&create_info, None)? };

        Ok(Self {
            _wayland: wayland,
            handle,
            loader,
        })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self) {
        self.loader.destroy_surface(self.handle, None);
    }
}

struct SwapchainImage {
    image: vk::Image,
    view: vk::ImageView,
}

impl SwapchainImage {
    fn new(device: &Device, image: vk::Image, format: vk::Format) -> Result<Self> {
        let view = unsafe {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let view_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(*subresource_range);

            device.handle.create_image_view(&view_info, None)?
        };

        Ok(SwapchainImage { image, view })
    }
}

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    loader: khr::Swapchain,

    present_mode: vk::PresentModeKHR,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,

    images: Vec<SwapchainImage>,
}

impl Swapchain {
    /// Create a new swapchain. `extent` is used to determine the size of the swapchain images only
    /// if it aren't able to determine it from `surface`.
    pub fn new(device: &Device, extent: vk::Extent2D) -> Result<Self> {
        let (surface_formats, present_modes, surface_caps) = unsafe {
            let format = device
                .surface
                .loader
                .get_physical_device_surface_formats(device.physical, device.surface.handle)?;
            let modes = device
                .surface
                .loader
                .get_physical_device_surface_present_modes(
                    device.physical,
                    device.surface.handle,
                )?;
            let caps = device
                .surface
                .loader
                .get_physical_device_surface_capabilities(device.physical, device.surface.handle)?;

            (format, modes, caps)
        };

        let queue_families = [device.graphics_queue.family_index];
        let min_image_count = 2.max(surface_caps.min_image_count);

        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == ash::vk::Format::B8G8R8A8_SRGB
                    && format.color_space == ash::vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| surface_formats.first())
            .ok_or_else(|| anyhow!("can't find valid surface format"))?
            .clone();

        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            vk::Extent2D {
                width: extent.width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: extent.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };

        let present_mode = present_modes
            .iter()
            .any(|mode| *mode == vk::PresentModeKHR::MAILBOX)
            .then(|| vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(device.surface.handle)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .image_extent(extent)
            .image_array_layers(1);

        let loader = khr::Swapchain::new(&device.instance, &device.handle);
        let handle = unsafe { loader.create_swapchain(&swapchain_info, None)? };

        let images = unsafe { loader.get_swapchain_images(handle)? };

        trace!("using {} swap chain images", images.len());

        let images: Result<Vec<_>> = images
            .into_iter()
            .map(|image| SwapchainImage::new(device, image, surface_format.format))
            .collect();

        Ok(Self {
            surface_format,
            images: images?,
            present_mode,
            handle,
            loader,
            extent,
        })
    }

    /// Recreate swapchain from `self` to a new `extent`.
    ///
    /// `extent` must be valid here unlike in `Self::new`, otherwise it could end in and endless
    /// cycle if recreating the swapchain, if for some reason the surface continues to give us and
    /// invalid extent.
    pub fn recreate(&mut self, device: &Device, extent: vk::Extent2D) -> Result<()> {
        if extent.width == u32::MAX {
            return Err(anyhow!("`extent` must be valid when recreating swapchain"));
        }

        let surface_caps = unsafe {
            device
                .surface
                .loader
                .get_physical_device_surface_capabilities(device.physical, device.surface.handle)?
        };

        let queue_families = [device.graphics_queue.family_index];
        let min_image_count = 2.max(surface_caps.min_image_count);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .old_swapchain(self.handle)
            .surface(device.surface.handle)
            .min_image_count(min_image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .image_extent(extent)
            .image_array_layers(1);

        let new = unsafe { self.loader.create_swapchain(&swapchain_info, None)? };

        unsafe {
            self.loader.destroy_swapchain(self.handle, None);

            for image in &self.images {
                device.handle.destroy_image_view(image.view, None);
            }
        }

        self.handle = new;

        let images = unsafe { self.loader.get_swapchain_images(self.handle)? };
        let images: Result<Vec<_>> = images
            .into_iter()
            .map(|image| SwapchainImage::new(device, image, self.surface_format.format))
            .collect();

        self.images = images?;

        Ok(())
    }

    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self, device: &Device) {
        self.loader.destroy_swapchain(self.handle, None);

        for image in &self.images {
            device.handle.destroy_image_view(image.view, None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct UniformData {
    perspective: Mat4,
    view: Mat4,
}

impl UniformData {
    fn update(&mut self, camera: &Camera) {
        self.perspective = camera.perspective;
        self.view = camera.view;
    }
}

pub struct UniformBuffers {
    /// One buffer for each frame.
    buffers: Buffers,
    /// Mapped memory of all the buffers.
    mapped: MappedMemory,
}

impl UniformBuffers {
    pub fn new(device: &Device) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(std::mem::size_of::<UniformData>() as u64);

        let create_infos = [create_info.build(); FrameQueue::FRAME_COUNT];
        let alignment = device.device_properties.limits.non_coherent_atom_size;

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let buffers = Buffers::new(device, &create_infos, memory_flags, alignment)?;
        let mapped = buffers.block.map_all(device)?;

        Ok(UniformBuffers { buffers, mapped })
    }

    fn upload_data(&self, frame_index: u64, data: &UniformData) {
        unsafe {
            let frames: &mut [UniformData] = self.mapped.as_slice();
            frames[frame_index as usize] = data.clone();
        }
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self, device: &Device) {
        self.buffers.destroy(device);
    }
}

pub fn create_shader_module(device: &Device, code: &[u8]) -> Result<vk::ShaderModule> {
    use std::mem::{align_of, size_of};

    if code.len() % size_of::<u32>() != 0 {
        return Err(anyhow!("shader code size must be a multiple of 4"));
    }
    if code.as_ptr().align_offset(align_of::<u32>()) != 0 {
        return Err(anyhow!("shader code must be aligned to `u32`"));
    }
    let code = unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u32, code.len() / 4) };
    let info = vk::ShaderModuleCreateInfo::builder().code(code);
    let module = unsafe { device.handle.create_shader_module(&info, None)? };

    Ok(module)
}

pub struct Pipeline {
    handle: vk::Pipeline,

    layout: vk::PipelineLayout,

    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,

    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl Pipeline {
    pub fn new_render(
        device: &Device,
        swapchain: &Swapchain,
        render_pass: &RenderPass,
        uniform_buffers: &UniformBuffers,
    ) -> Result<Self> {
        let vertex_module = create_shader_module(
            &device,
            include_bytes_aligned_as!(u32, "../shaders/vert.spv"),
        )?;

        let fragment_module = create_shader_module(
            &device,
            include_bytes_aligned_as!(u32, "../shaders/frag.spv"),
        )?;

        let entry = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(&entry)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module)
                .name(&entry)
                .build(),
        ];

        let vert_attrib = [vk::VertexInputAttributeDescription {
            format: vk::Format::R32G32B32_SFLOAT,
            binding: 0,
            location: 0,
            offset: 0,
        }];

        let vert_binding = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vert_attrib)
            .vertex_binding_descriptions(&vert_binding);

        let vert_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let descriptor_pool = unsafe {
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: FrameQueue::FRAME_COUNT as u32,
            }];

            let info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&sizes)
                .max_sets(swapchain.image_count());

            device.handle.create_descriptor_pool(&info, None)?
        };

        let descriptor_layout = unsafe {
            let layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .build()];

            let layout_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

            device
                .handle
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let descriptor_sets = unsafe {
            let layouts = vec![descriptor_layout; FrameQueue::FRAME_COUNT];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);

            device.handle.allocate_descriptor_sets(&alloc_info)?
        };

        for (set, buffer) in descriptor_sets
            .iter()
            .zip(uniform_buffers.buffers.buffers.iter())
        {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: buffer.handle,
                offset: 0,
                range: buffer.size(),
            }];
            let writes = [vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { device.handle.update_descriptor_sets(&writes, &[]) }
        }

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        }];

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterize_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .polygon_mode(vk::PolygonMode::FILL);

        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];

        let color_blend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

        let layout = unsafe {
            let layouts = [descriptor_layout];
            let info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);

            device.handle.create_pipeline_layout(&info, None)?
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vert_input_info)
            .input_assembly_state(&vert_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterize_info)
            .multisample_state(&multisample_info)
            .color_blend_state(&color_blend_info)
            .layout(layout)
            .render_pass(render_pass.handle)
            .subpass(0);

        let handle = unsafe {
            *device
                .handle
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .map_err(|(_, error)| anyhow!("failed to create pipeline: {error}"))?
                .first()
                .unwrap()
        };

        unsafe {
            device.handle.destroy_shader_module(vertex_module, None);
            device.handle.destroy_shader_module(fragment_module, None);
        }

        Ok(Self {
            handle,
            layout,
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
        })
    }

    /// Destroy and leave `self` in an invalid state.
    ///
    /// # Safety
    ///
    /// Don't use `self` after calling this function.
    unsafe fn destroy(&self, device: &Device) {
        device.handle.destroy_pipeline(self.handle, None);
        device.handle.destroy_pipeline_layout(self.layout, None);
        device
            .handle
            .destroy_descriptor_set_layout(self.descriptor_layout, None);
        device
            .handle
            .destroy_descriptor_pool(self.descriptor_pool, None);
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut ffi::c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*cb_data).p_message);

    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;

    if severity.contains(Severity::ERROR) {
        error!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::WARNING) {
        warn!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::INFO) {
        info!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::VERBOSE) {
        trace!("vulkan({ty:?}): {message:?}");
    }

    vk::FALSE
}
