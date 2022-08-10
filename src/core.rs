use anyhow::Result;
use ash::extensions::{ext, khr};
use ash::vk;
use glam::UVec3;
use smallvec::SmallVec;
use arrayvec::ArrayVec;

use std::ffi::{self, CStr, CString};
use std::{iter, mem, ops, slice};
use std::rc::Rc;
use std::cell::Cell;

use crate::handle::Handle;
use crate::resource::{self, Image, Buffer, TextureSampler};

pub struct Renderer {
    pub device: Device,
    pub swapchain: Swapchain,
    render_pass: RenderPass,
    frame_queue: FrameQueue,
    render_targets: RenderTargets,
}

impl Renderer {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let device = Device::new(window)?;
        let window_extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        let swapchain = Swapchain::new(&device, window_extent)?;
        let render_targets = RenderTargets::new(&device, &swapchain)?;
        let render_pass = RenderPass::new(&device, &swapchain, &render_targets)?;
        let frame_queue = FrameQueue::new(&device)?;

        device.wait_until_idle();

        Ok(Self {
            device,
            swapchain,
            render_pass,
            frame_queue,
            render_targets,
        })
    }

    pub fn draw<P, R>(&self, pre: P, render: R) -> Result<()>
    where
        P: FnOnce(&CommandRecorder),
        R: FnOnce(&CommandRecorder),
    {
        self.frame_queue.next_frame();

        let frame = self.frame_queue.current_frame();
        unsafe {
            let reset_flags = vk::CommandBufferResetFlags::empty();

            self.device.handle.wait_for_fences(&[frame.ready_to_draw], true, u64::MAX)?;
            self.device.handle.reset_command_buffer(frame.command_buffer, reset_flags)?;
        }

        loop {
            use NextSwapchainImage::*;

            let UpToDate { image_index } = self.swapchain.get_next_image(&frame)? else {
                // TODO: Do something here.
                panic!("out of date swapchain");
            };

            unsafe { self.device.handle.reset_fences(&[frame.ready_to_draw])?; }

            let recorder = CommandRecorder::new(
                self.device.handle.clone(),
                frame.index,
                frame.command_buffer.clone(),
            )?;

            pre(&recorder);

            let framebuffer =
                self.render_pass.get_framebuffer(&self.swapchain, frame.index, image_index);

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.2, 0.2, 0.2, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.00, 0.00, 0.00, 1.0],
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(self.render_pass.handle)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .clear_values(&clear_values);

            unsafe {
                self.device.handle.cmd_begin_render_pass(
                    recorder.buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                let viewports = self.swapchain.viewports();
                let scissors = self.swapchain.scissors();

                self.device.handle.cmd_set_viewport(recorder.buffer, 0, &viewports);
                self.device.handle.cmd_set_scissor(recorder.buffer, 0, &scissors);
            }

            render(&recorder);

            unsafe { self.device.handle.cmd_end_render_pass(recorder.buffer) };

            let command_buffer = recorder.end()?;

            // Submit command buffer to be rendered. Wait for semaphore `frame.presented` first and
            // signals `frame.rendered´ and `frame.ready_to_draw` when all commands have been
            // executed.
            unsafe {
                let wait = [frame.presented];
                let signals = [frame.rendered];
                let command_buffers = [command_buffer];
                let stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

                let submit_info = [vk::SubmitInfo::builder()
                    .wait_dst_stage_mask(&stages)
                    .wait_semaphores(&wait)
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&signals)
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

                self.swapchain.loader
                    .queue_present(self.device.transfer_queue.handle, &present_info)
                    .unwrap();
            }

            break;
        }

        Ok(())
    }

    /// Handle window resize. The extent of the swapchain and framebuffers will we match that of
    /// `window`.
    pub fn resize(&mut self, window: &winit::window::Window) -> Result<()> {
        trace!("resizing");

        self.device.wait_until_idle();

        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        self.swapchain.recreate(extent)?;
        self.render_targets = RenderTargets::new(&self.device, &self.swapchain)?;
        self.render_pass = RenderPass::new(&self.device, &self.swapchain, &self.render_targets)?;

        self.device.wait_until_idle();

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_until_idle();
    }
}

/// The device and data connected to the device used for rendering. This struct data is static
/// after creation and doesn't depend on external factors such as display size.
pub struct DeviceHandle {
    /// Name of `physical`, for instance "GTX 770".
    #[allow(dead_code)]
    name: String,

    pub handle: ash::Device,

    #[allow(unused)]
    entry: ash::Entry,

    instance: ash::Instance,
    physical: vk::PhysicalDevice,

    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_properties: vk::PhysicalDeviceProperties,

    surface: Surface,

    #[allow(dead_code)]
    messenger: DebugMessenger,

    /// Queue used to submit render commands.
    graphics_queue: Queue,

    /// Queue used to submut transfer commands, may be the same as `graphics_queue`.
    transfer_queue: Queue,

    /// Graphics pool fore recording render commands.
    graphics_pool: vk::CommandPool,

    /// Command pool for recording transfer commands. Not used for any rendering.
    transfer_pool: vk::CommandPool,
}

#[derive(Clone)]
pub struct Device {
    shared: Rc<DeviceHandle>,
}

impl ops::Deref for Device {
    type Target = DeviceHandle;
    
    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl Device {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        Ok(Self { shared: Rc::new(DeviceHandle::new(window)?) })
    }
}

impl DeviceHandle {
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
            let engine_name = CString::new("spillemotor").unwrap();
            let app_name = CString::new("spillemotor").unwrap();

            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 0, 0, 1))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 0, 0, 1))
                .api_version(version);

            let ext_names = [
                ext::DebugUtils::name().as_ptr(),
                khr::Surface::name().as_ptr(),
                #[cfg(target_os = "windows")]
                khr::Win32Surface::name().as_ptr(),
                #[cfg(target_os = "linux")]
                khr::WaylandSurface::name().as_ptr(),
                #[cfg(target_os = "macos")]
                ext::MetalSurface::name().as_ptr(),
                #[cfg(target_os = "macos")]
                vk::KhrPortabilityEnumerationFn::name().as_ptr(),
                #[cfg(target_os = "macos")]
                vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
            ];

            let flags = if cfg!(target_os = "macos") {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            /*
            let mut validation_features = vk::ValidationFeaturesEXT::builder()
                .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::DEBUG_PRINTF]);
            */

            let info = vk::InstanceCreateInfo::builder()
                .flags(flags)
                .push_next(&mut debug_info)
                // .push_next(&mut validation_features)
                .application_info(&app_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&ext_names);

            entry.create_instance(&info, None)?
        };

        let surface = Surface::new(&instance, &entry, &window)?;
        let messenger = DebugMessenger::new(&entry, &instance, &debug_info)?;

        // Select a physical device from heuristics. TODO: Improve this.
        let physical = unsafe {
            instance
                .enumerate_physical_devices()?
                .into_iter()
                .max_by_key(|dev| {
                    let properties = instance.get_physical_device_properties(*dev);

                    let name = CStr::from_ptr(properties.device_name.as_ptr())
                        .to_str()
                        .unwrap_or("invalid")
                        .to_string();

                    trace!("device candicate: {name}");

                    let mut score = properties.limits.max_image_dimension2_d;

                    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        score += 1000;
                    }

                    if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                        score += 500;
                    }

                    if properties.device_type == vk::PhysicalDeviceType::CPU {
                        score = 0;
                    }

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

        let extensions = [
            khr::Swapchain::name().as_ptr(),
            // vk::KhrShaderNonSemanticInfoFn::name().as_ptr(),
            #[cfg(target_os = "macos")]
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ];

        let enabled_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

        // Only create queues for both transfer and graphics if they aren't the same queue.
        let device_info = if transfer_index != graphics_index {
            vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&layer_names)
                .enabled_features(&enabled_features)
        } else {
            vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos[1..])
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&layer_names)
                .enabled_features(&enabled_features)
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

    /// Record and submut a command seqeunce to `Self::transfer_queue`. This will be performed
    /// immediately, meaning that the transfer will be done before this function returns.
    pub fn transfer_with<F, R>(&self, func: F) -> Result<R>
    where
        F: FnOnce(&CommandRecorder) -> R
    {
        // TODO: Avoid creating a new command buffer each time.
        let buffers = unsafe {
            let info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.transfer_pool)
                .command_buffer_count(1);
            self.handle.allocate_command_buffers(&info)?
        };

        let recorder = CommandRecorder::new(self.handle.clone(), 0, buffers[0])?;
        let ret = func(&recorder);

        let buffers = [recorder.buffer];
        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&buffers)
            .build()];

        unsafe {
            self.handle.end_command_buffer(recorder.buffer)?;
            self.handle.queue_submit(
                self.transfer_queue.handle,
                &submit_infos,
                vk::Fence::null(),
            )?;

            self.handle.queue_wait_idle(self.transfer_queue.handle)?;
            self.handle.free_command_buffers(self.transfer_pool, &buffers);
        }

        Ok(ret)
    }

    /// Get the sample count for msaa. For now it just the highest sample count the device
    /// supports, but below 8 samples.
    fn sample_count(&self) -> vk::SampleCountFlags {
        let counts = self
            .device_properties
            .limits
            .framebuffer_depth_sample_counts;

        // We don't wan't more than 8.
        let types = [
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ];

        for t in types {
            if counts.contains(t) {
                return t;
            }
        }

        return vk::SampleCountFlags::TYPE_1;
    }

    pub fn wait_until_idle(&self) {
        unsafe {
            self.handle.device_wait_idle().expect("failed waiting for idle device");
        }
    }
}

impl Drop for DeviceHandle {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_command_pool(self.graphics_pool, None);
            self.handle.destroy_command_pool(self.transfer_pool, None);
            self.handle.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// State relating to the printing vulkan debug info.
struct DebugMessenger {
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
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_debug_utils_messenger(self.handle, None); }
    }
}

struct RenderPass {
    handle: vk::RenderPass,
    /// We choose the amount of framebuffers to be the swapchain image count times `FRAMES_IN_FLIGHT`.
    ///
    /// This is to avoid allocating attachments for each swapchain image. Instead we have a
    /// framebuffer for each combination of frame resources and swapchain image, and can therefore
    /// have a render attachment for each frame in flight instead. It's essentially a mapping from
    /// swapchain image index and frame index to framebuffer.
    ///
    /// Alternatively you could either have a render attachments for each swapchain image, or
    /// create a new framebuffer dynamically before each frame. They both have disadvanges however.
    ///
    /// Some graphics cards gives up to 5 swapchain images it seems, so having 5 depth images and
    /// msaa images is be somewhat wasteful considering we only use `FRAMES_IN_FLIGHT` at a time.
    ///
    /// Creating new a new framebuffer each frame doesn't seem to be expensive as such, but it's
    /// still seems like an unnecessary cost to pay.
    ///
    /// # TODO
    ///
    /// Maybe use dynamic rendering instead.
    ///
    /// # Example
    ///
    /// Let's say the system gives 4 swapchain images and we have 2 images in flight, the
    /// framebuffers looks as follows.
    ///
    /// | Framebuffer | Frame | Image |
    /// |-------------|-------|-------|
    /// | 0           | 0     | 0     |
    /// | 1           | 0     | 1     |
    /// | 2           | 0     | 2     |
    /// | 3           | 0     | 3     |
    /// | 4           | 1     | 0     |
    /// | 5           | 1     | 1     |
    /// | 6           | 1     | 2     |
    /// | 7           | 1     | 3     |
    ///
    /// As you can see there is a framebuffer for each combination of frame and image.
    ///
    framebuffers: Vec<vk::Framebuffer>,

    device: Device,
}

impl RenderPass {
    fn new(device: &Device, swapchain: &Swapchain, render_targets: &RenderTargets) -> Result<Self> {
        let attachments = [
            // Swapchain image.
            vk::AttachmentDescription::builder()
                .format(swapchain.surface_format.format)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .samples(render_targets.sample_count)
                .build(),
            // Depth image.
            vk::AttachmentDescription::builder()
                .format(DEPTH_IMAGE_FORMAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .samples(render_targets.sample_count)
                .build(),
            // Msaa image.
            vk::AttachmentDescription::builder()
                .format(swapchain.surface_format.format)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
        ];

        let color_attachments = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let depth_attachment = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let msaa_attachments = [vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpasses = [vk::SubpassDescription::builder()
            .depth_stencil_attachment(&depth_attachment)
            .resolve_attachments(&msaa_attachments)
            .color_attachments(&color_attachments)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let subpass_dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_subpass(0)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build()];

        let handle = unsafe {
            let info = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&subpass_dependencies);
            device.handle.create_render_pass(&info, None)?
        };

        let framebuffers: Result<Vec<_>> = render_targets
            .iter()
            .flat_map(|t| iter::repeat(t).take(swapchain.image_count() as usize))
            .zip(swapchain.images.iter().cycle())
            .map(|(render_targets, swapchain_image)| {
                let views = [
                    render_targets.msaa.view,
                    render_targets.depth.view,
                    swapchain_image.view,
                ];
                let info = vk::FramebufferCreateInfo::builder()
                    .render_pass(handle)
                    .attachments(&views)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);
                Ok(unsafe { device.handle.create_framebuffer(&info, None)? })
            })
            .collect();

        Ok(RenderPass { device: device.clone(), framebuffers: framebuffers?, handle })
    }

    /// Get the framebuffer mapped to `image_index` and `frame_index`.
    ///
    /// See [`RenderPass::framebuffers`] for details.
    fn get_framebuffer(
        &self,
        swapchain: &Swapchain,
        frame_index: usize,
        image_index: u32,
    ) -> vk::Framebuffer {
        let index = frame_index as u32 * swapchain.image_count() + image_index;
        self.framebuffers[index as usize]
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_render_pass(self.handle, None);

            for framebuffer in &self.framebuffers {
                self.device.handle.destroy_framebuffer(*framebuffer, None);
            }
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

struct Frame {
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
    pub command_buffer: vk::CommandBuffer,

    pub index: usize,
}

impl Frame {
    fn new(device: &Device, index: usize, command_buffer: vk::CommandBuffer) -> Result<Self> {
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
            index,
        })
    }
}

struct FrameQueue {
    frames: Vec<Frame>,
    /// The index of the frame currently being rendered or presented. It changes just before
    /// rendering of the next image begins.
    frame_index: Cell<usize>,

    device: Device,
}

impl FrameQueue {
    pub fn new(device: &Device) -> Result<Self> {
        let command_buffers = unsafe {
            let command_buffer_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(device.graphics_pool)
                .command_buffer_count(FRAMES_IN_FLIGHT as u32);
            device.handle.allocate_command_buffers(&command_buffer_info)?
        };
        let frames: Result<Vec<_>> = command_buffers
            .into_iter()
            .enumerate()
            .map(|(i, buffer)| Frame::new(device, i, buffer))
            .collect();
        let frame_index = Cell::new(0_usize);
        Ok(Self { frames: frames?, frame_index, device: device.clone() })
    }

    pub fn next_frame(&self) {
        self.frame_index.set((self.index() + 1) % FRAMES_IN_FLIGHT);
    }

    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.index()]
    }

    pub fn index(&self) -> usize {
        self.frame_index.get()
    }
}

impl Drop for FrameQueue {
    fn drop(&mut self) {
        unsafe {
            let command_buffers: Vec<_> = self.frames
                .iter()
                .map(|frame| {
                    self.device.handle.destroy_semaphore(frame.rendered, None);
                    self.device.handle.destroy_semaphore(frame.presented, None);
                    self.device.handle.destroy_fence(frame.ready_to_draw, None);

                    frame.command_buffer
                })
                .collect();
            self.device.handle.free_command_buffers(self.device.graphics_pool, &command_buffers);
        }
    }
}

/// TODO: Make this handle more display servers besides Wayland.
struct Surface {
    handle: vk::SurfaceKHR,
    loader: khr::Surface,
}

impl Surface {
    fn new(
        instance: &ash::Instance,
        entry: &ash::Entry,
        window: &winit::window::Window,
    ) -> Result<Self> {
        let loader = khr::Surface::new(&entry, &instance);

        use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
        let handle = match window.raw_window_handle() {
            #[cfg(target_os = "windows")]
            RawWindowHandle::Win32(handle) => {
                let info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(handle.hinstance)
                    .hwnd(handle.hwnd);
                let loader = khr::Win32Surface::new(entry, instance);
                unsafe { loader.create_win32_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Wayland(handle) => {
                let info = vk::WaylandSurfaceCreateInfoKHR::builder()
                    .display(handle.display)
                    .surface(handle.surface);
                let loader = khr::WaylandSurface::new(entry, instance);
                unsafe { loader.create_wayland_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Xlib(handle) => {
                let info = vk::XlibSurfaceCreateInfoKHR::builder()
                    .dpy(handle.display as *mut _)
                    .window(handle.window);
                let loader = khr::XlibSurface::new(entry, instance);
                unsafe { loader.create_xlib_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Xcb(handle) => {
                let info = vk::XcbSurfaceCreateInfoKHR::builder()
                    .connection(handle.connection)
                    .window(handle.window);
                let loader = khr::XcbSurface::new(entry, instance);
                unsafe { loader.create_xcb_surface(&info, None) }
            }

            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(handle) => unsafe {
                use raw_window_metal::{appkit, Layer};

                let layer = appkit::metal_layer_from_handle(handle);
                let layer = match layer {
                    Layer::Existing(layer) | Layer::Allocated(layer) => layer as *mut _,
                    Layer::None => {
                        return Err(anyhow!("failed to load metal layer"));
                    }
                };

                let info = vk::MetalSurfaceCreateInfoEXT::builder().layer(&*layer);
                let loader = ext::MetalSurface::new(entry, instance);

                loader.create_metal_surface(&info, None)
            },
            _ => {
                return Err(anyhow!("unsupported platform"));
            }
        };
        Ok(Self { handle: handle?, loader })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.handle, None) }
    }
}

struct SwapchainImage {
    #[allow(dead_code)]
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

#[derive(Clone)]
struct RenderTarget<'a> {
    msaa: &'a Image,
    depth: &'a Image,
}

struct RenderTargets {
    /// One depth and MSAA image per frame.
    ///
    /// | Frame | Image |
    /// |-------|-------|
    /// | 0     | depth |
    /// | 0     | msaa  |
    /// | 1     | depth |
    /// | 1     | msaa  |
    ///
    images: Vec<Rc<Image>>,
    sample_count: vk::SampleCountFlags,
}

impl RenderTargets {
    fn new(device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let queue_families = [device.graphics_queue.family_index];

        let extent = vk::Extent3D {
            width: swapchain.extent.width,
            height: swapchain.extent.height,
            depth: 1,
        };

        let sample_count = device.sample_count();

        let depth_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .format(DEPTH_IMAGE_FORMAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(sample_count);
        let depth_subresource_info = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let depth_view_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(DEPTH_IMAGE_FORMAT)
            .subresource_range(depth_subresource_info);
        let msaa_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .usage(
                vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .format(swapchain.surface_format.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(sample_count);
        let msaa_subresource_info = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let msaa_view_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(swapchain.surface_format.format)
            .subresource_range(msaa_subresource_info);
        let image_infos: Vec<_> = [depth_image_info.build(), msaa_image_info.build()]
            .iter()
            .cycle()
            .map(|info| *info)
            .take(FRAMES_IN_FLIGHT * 2)
            .collect();
        let view_infos: Vec<_> = [depth_view_info.build(), msaa_view_info.build()]
            .iter()
            .cycle()
            .map(|info| *info)
            .take(FRAMES_IN_FLIGHT * 2)
            .collect();
        let (images, _) = resource::create_images_raw(
            device,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &image_infos,
            &view_infos,
        )?;

        Ok(Self { images, sample_count })
    }

    /// Iterator over each render target.
    fn iter(&self) -> impl Iterator<Item = RenderTarget> {
        self.images.chunks(2).map(|images| RenderTarget { depth: &images[0], msaa: &images[1] })
    }
}

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    loader: khr::Swapchain,

    present_mode: vk::PresentModeKHR,
    surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,

    images: Vec<SwapchainImage>,

    device: Device,
}

enum NextSwapchainImage {
    UpToDate {
        image_index: u32,
    },
    OutOfDate,
}

impl Swapchain {
    /// Create a new swapchain. `extent` is used to determine the size of the swapchain images only
    /// if it aren't able to determine it from `surface`.
    pub fn new(device: &Device, extent: vk::Extent2D) -> Result<Self> {
        let (surface_formats, _present_modes, surface_caps) = unsafe {
            let format = device
                .surface
                .loader
                .get_physical_device_surface_formats(
                    device.physical,
                    device.surface.handle,
                )?;
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
                .get_physical_device_surface_capabilities(
                    device.physical,
                    device.surface.handle,
                )?;
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
        let present_mode = _present_modes
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
            device: device.clone(),
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
    pub fn recreate(&mut self, extent: vk::Extent2D) -> Result<()> {
        if extent.width == u32::MAX {
            return Err(anyhow!("`extent` must be valid when recreating swapchain"));
        }

        let surface_caps = unsafe {
            self.device.surface.loader.get_physical_device_surface_capabilities(
                self.device.physical,
                self.device.surface.handle,
            )?
        };

        let queue_families = [self.device.graphics_queue.family_index];
        let min_image_count = (FRAMES_IN_FLIGHT as u32).max(surface_caps.min_image_count);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .old_swapchain(self.handle)
            .surface(self.device.surface.handle)
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
                self.device.handle.destroy_image_view(image.view, None);
            }
        }

        self.handle = new;
        self.extent = extent;

        let images = unsafe { self.loader.get_swapchain_images(self.handle)? };
        let images: Result<Vec<_>> = images
            .into_iter()
            .map(|image| SwapchainImage::new(&self.device, image, self.surface_format.format))
            .collect();

        self.images = images?;

        Ok(())
    }

    fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    fn get_next_image(&self, frame: &Frame) -> Result<NextSwapchainImage> {
        let next_image = unsafe {
            self.loader.acquire_next_image(
                self.handle,
                u64::MAX,
                frame.presented,
                vk::Fence::null(),
            )
        };
        match next_image {
            Ok((image_index, false)) => Ok(NextSwapchainImage::UpToDate { image_index }),
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                Ok(NextSwapchainImage::OutOfDate)
            }
            Err(result) => Err(result.into()),
        }
    }

    pub fn viewports(&self) -> [vk::Viewport; 1] {
        [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }]
    }

    pub fn scissors(&self) -> [vk::Rect2D; 1] {
        [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        }]
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.extent.width as f32 / self.extent.height as f32
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.handle, None);
            
            for image in &self.images {
                self.device.handle.destroy_image_view(image.view, None);
            }
        }
    }
}

pub struct ShaderModule {
    handle: vk::ShaderModule,
    entry: CString,

    device: Device,
}

impl ShaderModule {
    pub fn new(
        renderer: &Renderer,
        entry: &str,
        code: &[u8],
    ) -> Result<Self> {
        let device = renderer.device.clone();

        if code.len() % mem::size_of::<u32>() != 0 {
            return Err(anyhow!("shader code size must be a multiple of 4"));
        }

        if code.as_ptr().align_offset(mem::align_of::<u32>()) != 0 {
            return Err(anyhow!("shader code must be aligned to `u32`"));
        }

        let code = unsafe { slice::from_raw_parts(code.as_ptr() as *const u32, code.len() / 4) };
        let info = vk::ShaderModuleCreateInfo::builder().code(code);
        let handle = unsafe { device.handle.create_shader_module(&info, None)? };

        let Ok(entry) = CString::new(entry) else {
            return Err(anyhow!("invalid entry name `entry`"));
        };

        Ok(ShaderModule { device, handle, entry })
    }

    fn stage_create_info(
        &self,
        stage: vk::ShaderStageFlags,
    ) -> impl ops::Deref<Target = vk::PipelineShaderStageCreateInfo> + '_ {
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(stage)
            .module(self.handle)
            .name(&self.entry)
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_shader_module(self.handle, None);
        }
    }
}

pub struct LayoutBinding {
    pub stage: vk::ShaderStageFlags,
    pub ty: vk::DescriptorType,
}

struct LayoutBindings {
    bindings: SmallVec<[vk::DescriptorSetLayoutBinding; 6]>,
}

impl LayoutBindings {
    fn new(bindings: &[LayoutBinding]) -> Self  {
        let bindings = bindings
            .iter()
            .enumerate()
            .map(|(i, binding)| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(i as u32)
                    .descriptor_type(binding.ty)
                    .descriptor_count(1)
                    .stage_flags(binding.stage)
                    .build()
            })
            .collect();
        Self { bindings }
    }

    fn iter(&self) -> impl Iterator<Item = &vk::DescriptorSetLayoutBinding> {
        self.bindings.iter()
    }
}

pub enum DescriptorBinding<const N: usize> {
    Buffer([Rc<Buffer>; N]),
    Image(Handle<TextureSampler>, [Rc<Image>; N]),
}

#[derive(Clone)]
pub struct DescriptorSetLayout {
    pub handle: vk::DescriptorSetLayout,
    bindings: Rc<LayoutBindings>,
    device: Device, 
}

impl DescriptorSetLayout {
    pub fn new(renderer: &Renderer, bindings: &[LayoutBinding]) -> Result<Handle<Self>> {
        Self::create(renderer, LayoutBindings::new(bindings))
    }

    fn create(renderer: &Renderer, bindings: LayoutBindings) -> Result<Handle<Self>> {
        let device = renderer.device.clone();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings.bindings)
            .build();
        let handle = unsafe { device.handle.create_descriptor_set_layout(&layout_info, None)? };
        Ok(Handle::new(Self {handle, bindings: Rc::new(bindings), device }))
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_descriptor_set_layout(self.handle, None); }
    }
}

enum BoundResource {
    Sampler(Handle<TextureSampler>),
    Buffer(Rc<Buffer>),
    Image(Rc<Image>),
}

#[derive(Clone)]
pub struct DescriptorSet {
    pub layout: Handle<DescriptorSetLayout>,

    pub pool: vk::DescriptorPool,
    pub sets: ArrayVec<vk::DescriptorSet, FRAMES_IN_FLIGHT>,

    #[allow(dead_code)]
    resources: Rc<SmallVec<[BoundResource; 32]>>,
    device: Device,
}

impl ops::Index<usize> for DescriptorSet {
    type Output = vk::DescriptorSet;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.sets[idx % self.sets.len()]
    }
}

impl DescriptorSet {
    pub fn new_single(
        renderer: &Renderer,
        layout: Handle<DescriptorSetLayout>,
        bindings: &[DescriptorBinding<1>],
    ) -> Result<Self> {
        Self::new(renderer, layout, bindings)
    }

    pub fn new_per_frame(
        renderer: &Renderer,
        layout: Handle<DescriptorSetLayout>,
        bindings: &[DescriptorBinding<FRAMES_IN_FLIGHT>],
    ) -> Result<Self> {
        Self::new(renderer, layout, bindings)
    }

    fn new<const N: usize>(
        renderer: &Renderer,
        layout: Handle<DescriptorSetLayout>,
        bindings: &[DescriptorBinding<N>],
    ) -> Result<Self> {
        let device = renderer.device.clone();

        let pool = unsafe {
            let mut sizes = Vec::<vk::DescriptorPoolSize>::default();
            
            for layout_binding in layout.bindings.iter() {
                match sizes.iter_mut().position(|size| size.ty == layout_binding.descriptor_type) {
                    Some(pos) => sizes[pos].descriptor_count += N as u32,
                    None => {
                        sizes.push(vk::DescriptorPoolSize {
                            ty: layout_binding.descriptor_type,
                            descriptor_count: N as u32,
                        });
                    }
                }
            }

            let info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&sizes)
                .max_sets(N as u32);

            device.handle.create_descriptor_pool(&info, None)?
        };

        let sets = unsafe {
            let layouts: SmallVec<[_; 12]> = std::iter::repeat(layout.clone())
                .take(N)
                .map(|layout| layout.handle)
                .collect();
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            device.handle.allocate_descriptor_sets(&alloc_info)?
        };

        for (n, set) in sets.iter().enumerate() {
            struct Info {
                ty: vk::DescriptorType,
                buffer: [vk::DescriptorBufferInfo; 1],
                image: [vk::DescriptorImageInfo; 1],
            }

            let infos: SmallVec<[Info; 12]> = bindings
                .iter()
                .zip(layout.bindings.iter())
                .map(|(binding, layout_binding)| {
                    match &binding {
                        DescriptorBinding::Buffer(buffers) => Info {
                            ty: layout_binding.descriptor_type,
                            image: Default::default(),
                            buffer: [vk::DescriptorBufferInfo {
                                buffer: buffers[n].handle,
                                offset: 0,
                                range: buffers[n].size(),
                            }],
                        },
                        DescriptorBinding::Image(sampler, images) => Info {
                            ty: layout_binding.descriptor_type,
                            buffer: Default::default(),
                            image: [vk::DescriptorImageInfo {
                                image_layout: images[n].layout.get(),
                                image_view: images[n].view,
                                sampler: sampler.handle,
                            }],
                        },
                    }
                })
                .collect();

            let writes: SmallVec<[vk::WriteDescriptorSet; 12]> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(*set)
                        .dst_binding(binding as u32)
                        .descriptor_type(info.ty)
                        .buffer_info(&info.buffer)
                        .image_info(&info.image)
                        .build()
                })
                .collect();

            unsafe { device.handle.update_descriptor_sets(&writes, &[]) }
        }

        let mut resources: SmallVec<[_; 32]> = SmallVec::default();
        for binding in bindings {
            match &binding {
                DescriptorBinding::Buffer(buffers) => {
                    for buffer in buffers {
                        resources.push(BoundResource::Buffer(buffer.clone()));
                    }
                }
                DescriptorBinding::Image(sampler, images) => {
                    resources.push(BoundResource::Sampler(sampler.clone()));

                    for image in images {
                        resources.push(BoundResource::Image(image.clone()));
                    }
                }
            }
        }

        let resources = Rc::new(resources);

        let sets = {
            let mut array = ArrayVec::default();
            for set in sets.into_iter() {
                array.push(set);
            }
            array
        };

        Ok(Self { device, layout, pool, sets, resources })
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_descriptor_pool(self.pool, None); }
    }
}

#[derive(Clone)]
pub struct PipelineLayout {
    pub handle: vk::PipelineLayout,

    // TODO: Make Rc.
    #[allow(dead_code)]
    descriptor_layouts: SmallVec<[Handle<DescriptorSetLayout>; 2]>,

    device: Device,
}

impl PipelineLayout {
    pub fn new(
        renderer: &Renderer,
        consts: &[vk::PushConstantRange],
        layouts: &[Handle<DescriptorSetLayout>],
    ) -> Result<Handle<Self>> {
        let device = renderer.device.clone();

        let handle = unsafe {
            let layouts: SmallVec<[_; 12]> = layouts
                .iter()
                .map(|layout| layout.handle)
                .collect();
            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&consts);
            device.handle.create_pipeline_layout(&info, None)?
        };

        let mut descriptor_layouts = SmallVec::default();
        for layout in layouts {
            descriptor_layouts.push(layout.clone());
        }

        Ok(Handle::new(Self { handle, descriptor_layouts, device }))
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_pipeline_layout(self.handle, None); }
    }
}

pub struct ComputePipeline {
    handle: vk::Pipeline,
    layout: Handle<PipelineLayout>,

    device: Device,
}

impl ComputePipeline {
    pub fn new(
        renderer: &Renderer,
        layout: Handle<PipelineLayout>,
        shader: &ShaderModule,
    ) -> Result<Self> {
        let device = renderer.device.clone();
        let stage = shader.stage_create_info(vk::ShaderStageFlags::COMPUTE);
        let create_infos = [vk::ComputePipelineCreateInfo::builder()
            .layout(layout.handle)
            .stage(*stage)
            .build()];
        let handle = unsafe {
            device.handle
                .create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .map_err(|(_, err)| err)?
                .first()
                .unwrap()
                .clone()
        };
        Ok(Self { device, handle, layout })
    }

    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_pipeline(self.handle, None); }
    }
}

pub struct GraphicsPipelineReq<'a> {
    pub layout: Handle<PipelineLayout>,
    pub vertex_attributes: &'a [vk::VertexInputAttributeDescription],
    pub vertex_bindings: &'a [vk::VertexInputBindingDescription],
    pub depth_stencil_info: &'a vk::PipelineDepthStencilStateCreateInfo,
    pub vertex_shader: &'a ShaderModule,
    pub fragment_shader: &'a ShaderModule,
}

pub struct GraphicsPipeline {
    pub handle: vk::Pipeline,
    pub layout: Handle<PipelineLayout>,

    device: Device,
}

impl GraphicsPipeline {
    pub fn new(renderer: &Renderer, req: GraphicsPipelineReq) -> Result<Self> {
        let device = renderer.device.clone();

        let shader_stages = [
            *req.vertex_shader.stage_create_info(vk::ShaderStageFlags::VERTEX),
            *req.fragment_shader.stage_create_info(vk::ShaderStageFlags::FRAGMENT),
        ];

        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&req.vertex_attributes)
            .vertex_binding_descriptions(&req.vertex_bindings);

        let vert_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewports = renderer.swapchain.viewports();
        let scissors = renderer.swapchain.scissors();

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterize_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .polygon_mode(vk::PolygonMode::FILL);
        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(renderer.render_targets.sample_count);
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

        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .dynamic_state(&dynamic_state)
            .stages(&shader_stages)
            .vertex_input_state(&vert_input_info)
            .input_assembly_state(&vert_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterize_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&req.depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .layout(req.layout.handle)
            .render_pass(renderer.render_pass.handle)
            .subpass(0);

        let handle = unsafe {
            *device.handle
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .map_err(|(_, error)| anyhow!("failed to create pipeline: {error}"))?
                .first()
                .unwrap()
        };

        Ok(Self { device, handle, layout: req.layout })
    }

    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_pipeline(self.handle, None); }
    }
}

pub struct CommandRecorder {
    pub buffer: vk::CommandBuffer,
    frame_index: usize,

    // TODO: Make this `Device` when we do some refactoring.
    device: ash::Device,
}

impl CommandRecorder {
    fn new(device: ash::Device, frame_index: usize, buffer: vk::CommandBuffer) -> Result<Self> {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(buffer, &begin_info)?;
        }

        Ok(Self { buffer, frame_index, device: device.clone() })
    }

    fn end(self) -> Result<vk::CommandBuffer> {
        unsafe { self.device.end_command_buffer(self.buffer)?; }
        Ok(self.buffer)
    }

    pub fn frame_index(&self) -> usize {
        self.frame_index
    }

    pub fn copy_buffers(&self, src: &Buffer, dst: &Buffer) {
        let size = src.size().min(dst.size());
        let regions = [vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size)
            .build()];
        unsafe {
            self.device.cmd_copy_buffer(
                self.buffer,
                src.handle,
                dst.handle,
                &regions,
            );
        }
    }

    pub fn copy_buffer_to_image(&self, src: &Buffer, dst: &Image) {
        // NOTE: Make sure this is updated when adding mip levels.
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
            self.device.cmd_copy_buffer_to_image(
                self.buffer,
                src.handle,
                dst.handle,
                dst.layout(),
                &regions,
            );
        }
    }

    /// Transition the layout of `image` to `new`.
    ///
    /// For now it handles two transitions (format` references the current layout of the image):
    ///
    /// | `format`               | `new`                      |
    /// |------------------------|----------------------------|
    /// | `UNDEFINED`            | `TRANSFER_DST_OPTIMAL`     |
    /// | `TRANSFER_DST_OPTIMAL` | `SHADER_READ_ONLY_OPTIMAL` |
    ///
    /// The transition will fail if the transfer doesn't fit into the tabel.
    pub fn transition_image_layout(&self, image: &Image, new: vk::ImageLayout) {
        // NOTE: Make sure this is updated when adding mip levels.
        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let mut barrier = vk::ImageMemoryBarrier::builder()
            .image(image.handle)
            .old_layout(image.layout())
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource);

        let (src_stage, dst_stage) = match (image.layout(), new) {
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

        image.layout.set(new);

        unsafe {
            let barriers = [barrier.build()];
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
    }

    pub fn dispatch(&self, pipeline: &ComputePipeline, group_count: UVec3) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle,
            );
            self.device.cmd_dispatch(
                self.buffer,
                group_count.x,
                group_count.y,
                group_count.z,
            );
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: &PipelineLayout,
        descriptors: &[&DescriptorSet],
    ) {
        let descs: SmallVec<[_; 12]> = descriptors
            .iter()
            .map(|desc| desc[self.frame_index])
            .collect();
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.buffer,
                bind_point,
                layout.handle,
                0,
                &descs,
                &[],
            );
        }
    }

    pub fn bind_vertex_buffer(&self, buffer: &Buffer) {
        unsafe {
            self.device.cmd_bind_vertex_buffers(
                self.buffer, 0, &[buffer.handle], &[0],
            );
        }
    }

    pub fn bind_index_buffer(&self, buffer: &Buffer) {
        unsafe {
            self.device.cmd_bind_index_buffer(
                self.buffer, buffer.handle, 0, vk::IndexType::UINT16,
            );
        }
    }

    pub fn push_constants<T: bytemuck::NoUninit>(
        &self,
        layout: &PipelineLayout,
        stage: vk::ShaderStageFlags,
        offset: u32,
        val: &T,
    ) {
        let bytes = bytemuck::bytes_of(val);
        unsafe {
            self.device.cmd_push_constants(
                self.buffer,
                layout.handle,
                stage,
                offset,
                bytes,
            );
        }
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.device.cmd_bind_pipeline(
                self.buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.handle,
            );
        }
    }

    pub fn draw(&self, index_count: u32, index_start: u32, vertex_off: i32) {
        unsafe {
            self.device.cmd_draw_indexed(self.buffer, index_count, 1, index_start, vertex_off, 0);
        }
    }

    pub fn buffer_rw_barrier(
        &self,
        buffer: &Buffer,
        write_stage: vk::PipelineStageFlags,
        read_stage: vk::PipelineStageFlags,
    ) {
        let barriers = [vk::BufferMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer.handle)
            .offset(0)
            .size(buffer.size())
            .build()];
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                write_stage,
                read_stage,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
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

/// The number of frames being worked on concurrently. This could easily be higher, you could for
/// instance work on all swapchain images concurrently, but you risk having the CPU run ahead,
/// which add latency and delays. You also save some memory by having less depth buffers and such.
pub const FRAMES_IN_FLIGHT: usize = 2;

const DEPTH_IMAGE_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
