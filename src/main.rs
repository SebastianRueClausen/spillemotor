#[macro_use]
extern crate log;

#[macro_use]
extern crate anyhow;

#[macro_use]
mod macros;

mod camera;
mod resource;

use anyhow::Result;

use ash::extensions::{ext, khr};
use ash::vk;

use std::ffi::{self, CStr, CString};
use std::path::Path;
use std::time::{Duration, Instant};

use camera::{Camera, InputState};
use resource::{Buffer, Mesh, Vertex};

fn main() -> Result<()> {
    env_logger::init();

    use winit::event::{Event, WindowEvent, ElementState};
    use winit::event_loop::ControlFlow;

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    window.set_decorations(true);
    window.set_cursor_grab(true)?;
    window.set_cursor_visible(false);

    let mut minimized = false;

    let mut last_draw = Instant::now();
    let mut last_update = Instant::now();

    let mut ctx = Some(Context::new(&window)?);
    let mut input_state = InputState::default();

    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            ctx.take().map(Context::destroy);
            *controlflow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::KeyboardInput {
                input,
                ..
            },
            ..
        } => {
            if let Some(key) = input.virtual_keycode {
                match input.state {
                    ElementState::Pressed => input_state.key_pressed(key),
                    ElementState::Released => input_state.key_released(key),
                }
            }
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved {
                position,
                ..
            },
            ..
        } => {
            input_state.mouse_moved((position.x, position.y));
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            if size.width == 0 && size.height == 0 {
                minimized = true
            } else {
                minimized = false;
                if let Some(ctx) = &mut ctx {
                    ctx.recreate_swapchain(&window)
                        .expect("failed resizing window");
                }
            }
        }
        Event::RedrawRequested(_) => {
            if let Some(ctx) = &mut ctx {
                last_draw = Instant::now();

                if !minimized && !ctx.draw().expect("failed drawing to the screen") {
                    warn!("out of data surface when trying to draw");
                }
            }
        }
        Event::MainEventsCleared => {
            if let Some(ctx) = &mut ctx {
                ctx.camera.update(&mut input_state, last_update.elapsed());
                last_update = Instant::now();
            }

            if let Some(left) = Duration::from_millis(16).checked_sub(last_draw.elapsed()) {
                *controlflow = ControlFlow::WaitUntil(Instant::now() + left);
            } else {
                window.request_redraw();
            }
        }
        _ => {}
    });
}

struct WaylandSurface {
    _wayland: khr::WaylandSurface,
    surface: vk::SurfaceKHR,
    loader: khr::Surface,
}

impl WaylandSurface {
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

        let surface = unsafe { wayland.create_wayland_surface(&create_info, None)? };

        Ok(Self {
            _wayland: wayland,
            surface,
            loader,
        })
    }

    fn destroy(self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

struct Context {
    device_name: String,

    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,

    memory_properties: vk::PhysicalDeviceMemoryProperties,

    physical: vk::PhysicalDevice,

    graphics_queue: Queue,
    transfer_queue: Queue,

    frame_sync: Vec<FrameSync>,
    command_buffers: Vec<vk::CommandBuffer>,

    graphics_pool: vk::CommandPool,
    transfer_pool: vk::CommandPool,

    render_pass: vk::RenderPass,

    debug_loader: ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,

    surface: WaylandSurface,
    swapchain: Swapchain,

    current_frame: usize,

    swapchain_images: Vec<SwapchainImage>,

    pipeline: Pipeline,

    camera: Camera,
}

impl Context {
    fn new(window: &winit::window::Window) -> Result<Self> {
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

        let surface = WaylandSurface::new(&instance, &entry, window)?;

        let debug_loader = ext::DebugUtils::new(&entry, &instance);
        let messenger = unsafe { debug_loader.create_debug_utils_messenger(&debug_info, None)? };

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

        let device_name = unsafe {
            let properties = instance.get_physical_device_properties(physical);
            CStr::from_ptr(properties.device_name.as_ptr())
                .to_str()
                .unwrap_or("invalid")
                .to_string()
        };

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

        trace!("using device: {device_name}");

        let queue_props = unsafe { instance.get_physical_device_queue_family_properties(physical) };

        let graphics_index = queue_props
            .iter()
            .enumerate()
            .position(|(i, p)| {
                p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && unsafe {
                        surface
                            .loader
                            .get_physical_device_surface_support(
                                physical,
                                i as u32,
                                surface.surface,
                            )
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

        let device = unsafe { instance.create_device(physical, &device_info, None)? };

        let graphics_queue = Queue::new(&device, graphics_index);
        let transfer_queue = Queue::new(&device, transfer_index);

        let mut swapchain = Swapchain::new(
            &instance,
            &device,
            physical,
            &graphics_queue,
            &surface,
            vk::Extent2D {
                width: window.inner_size().width,
                height: window.inner_size().height,
            },
            None,
        )?;

        let aspect_ratio = swapchain.extent.width as f32 / swapchain.extent.height as f32;
        let camera = Camera::new(aspect_ratio);

        let render_pass = create_render_pass(&device, &swapchain)?;
        let pipeline = Pipeline::new(
            &device,
            &memory_properties,
            &swapchain,
            render_pass,
            &camera,
        )?;

        let graphics_pool = unsafe {
            let graphics_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(graphics_queue.index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            device.create_command_pool(&graphics_pool_info, None)?
        };

        let transfer_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(transfer_queue.index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let transfer_pool = unsafe { device.create_command_pool(&transfer_pool_info, None)? };

        let swapchain_images = swapchain.create_images(&device, render_pass)?;

        let command_buffers = unsafe {
            let command_buffer_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(graphics_pool)
                .command_buffer_count(swapchain_images.len() as u32);

            device.allocate_command_buffers(&command_buffer_info)?
        };

        record_command_buffers(
            &device,
            &pipeline,
            render_pass,
            &swapchain,
            &swapchain_images,
            &command_buffers,
        )?;

        let frame_sync = vec![FrameSync::new(&device)?, FrameSync::new(&device)?];


        Ok(Self {
            graphics_pool,
            transfer_pool,
            debug_loader,
            messenger,
            device_name,
            entry,
            instance,
            physical,
            memory_properties,
            device,
            graphics_queue,
            transfer_queue,
            render_pass,
            surface,
            swapchain,
            swapchain_images,
            frame_sync,
            current_frame: 0,
            pipeline,
            command_buffers,
            camera,
        })
    }

    fn destroy(mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("failed waiting for idle device");

            self.device
                .free_command_buffers(self.graphics_pool, &self.command_buffers);
            self.pipeline.destroy(&self.device);
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_images
                .drain(..)
                .for_each(|image| image.destroy(&self.device));
            self.swapchain.destroy();

            self.frame_sync
                .drain(..)
                .for_each(|sync| sync.destroy(&self.device));

            self.device.destroy_command_pool(self.graphics_pool, None);
            self.device.destroy_command_pool(self.transfer_pool, None);

            self.debug_loader
                .destroy_debug_utils_messenger(self.messenger, None);
            self.device.destroy_device(None);
            self.surface.destroy();
            self.instance.destroy_instance(None);
        }
    }

    fn draw(&mut self) -> Result<bool> {
        self.next_frame();

        let sync = self.current_frame_sync();

        // Wait for the frame to be ready to draw to.
        unsafe {
            self.device
                .wait_for_fences(&[sync.ready_to_draw], true, u64::MAX)?;
        }

        let next_image = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                sync.presented,
                vk::Fence::null(),
            )
        };

        match next_image {
            // Getting the next image succeded and the swapchain is optimal.
            Ok((image_index, false)) => {
                let command_buffer = self.command_buffers[image_index as usize];
                
                upload_camera_to_uniform_buffer(&self.camera, &self.pipeline.uniform_buffer);

                unsafe {
                    self.device.reset_fences(&[sync.ready_to_draw])?;
                }

                // Submit command buffer to rendered. Wait for semaphore `sync.presented` first and
                // signals `sync.rendered´ and `sync.ready_to_draw` when all commands have been
                // executed.
                unsafe {
                    let wait = [sync.presented];
                    let command_buffer = [command_buffer];
                    let signal = [sync.rendered];
                    let stage = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

                    let submit_info = [vk::SubmitInfo::builder()
                        .wait_dst_stage_mask(&stage)
                        .wait_semaphores(&wait)
                        .command_buffers(&command_buffer)
                        .signal_semaphores(&signal)
                        .build()];

                    self.device.queue_submit(
                        self.graphics_queue.queue,
                        &submit_info,
                        sync.ready_to_draw,
                    )?;
                }

                // Wait for the frame to be rendered before presenting it to the surface.
                unsafe {
                    let wait = [sync.rendered];
                    let swapchains = [self.swapchain.swapchain];
                    let indices = [image_index];

                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&wait)
                        .swapchains(&swapchains)
                        .image_indices(&indices);

                    self.swapchain
                        .loader
                        .queue_present(self.graphics_queue.queue, &present_info)
                        .unwrap();

                    Ok(true)
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Ok((_, true)) => Ok(false),
            Err(result) => Err(result.into()),
        }
    }

    fn next_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    fn current_frame_sync(&self) -> &FrameSync {
        &self.frame_sync[self.current_frame]
    }

    fn recreate_swapchain(&mut self, window: &winit::window::Window) -> Result<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("failed waiting for idle device");

            self.device
                .free_command_buffers(self.graphics_pool, &self.command_buffers);
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_images
                .drain(..)
                .for_each(|image| image.destroy(&self.device));
        }

        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        let aspect_ratio = extent.width as f32 / extent.height as f32;
        self.camera.update_perspective(aspect_ratio);

        self.swapchain = Swapchain::new(
            &self.instance,
            &self.device,
            self.physical,
            &self.graphics_queue,
            &self.surface,
            extent,
            Some(self.swapchain.clone()),
        )?;

        self.render_pass = create_render_pass(&self.device, &self.swapchain)?;
        self.swapchain_images = self
            .swapchain
            .create_images(&self.device, self.render_pass)?;

        self.command_buffers = unsafe {
            let command_buffer_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.graphics_pool)
                .command_buffer_count(self.swapchain_images.len() as u32);

            self.device.allocate_command_buffers(&command_buffer_info)?
        };

        record_command_buffers(
            &self.device,
            &self.pipeline,
            self.render_pass,
            &self.swapchain,
            &self.swapchain_images,
            &self.command_buffers,
        )?;

        Ok(())
    }
}

fn record_command_buffers(
    device: &ash::Device,
    pipeline: &Pipeline,
    render_pass: vk::RenderPass,
    swapchain: &Swapchain,
    swapchain_images: &[SwapchainImage],
    buffers: &[vk::CommandBuffer],
) -> Result<()> {
    for ((buffer, image), descriptor) in buffers
        .iter()
        .zip(swapchain_images.iter())
        .zip(pipeline.descriptor_sets.iter())
    {
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
            device.begin_command_buffer(*buffer, &begin_info)?;
        }

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.02, 0.02, 0.02, 1.0],
            },
        }];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(image.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(
                *buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(*buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                *buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[*descriptor],
                &[],
            );

            device.cmd_bind_vertex_buffers(*buffer, 0, &[pipeline.vertex_buffer.buffer], &[0]);
            device.cmd_draw(*buffer, pipeline.mesh.verts.len() as u32, 1, 0, 0);

            device.cmd_end_render_pass(*buffer);
            device.end_command_buffer(*buffer)?;
        }
    }

    Ok(())
}

fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
    use std::mem::{align_of, size_of};

    assert_eq!(
        code.len() % size_of::<u32>(),
        0,
        "code size must be multiple of 4"
    );
    assert_eq!(
        code.as_ptr().align_offset(align_of::<u32>()),
        0,
        "code must be aligned to `u32`"
    );

    let code = unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u32, code.len() / 4) };

    let info = vk::ShaderModuleCreateInfo::builder().code(code);

    Ok(unsafe { device.create_shader_module(&info, None)? })
}

fn create_render_pass(device: &ash::Device, swapchain: &Swapchain) -> Result<vk::RenderPass> {
    let attachments = [vk::AttachmentDescription::builder()
        .format(swapchain.format)
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

    let info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&subpass_dependencies);

    Ok(unsafe { device.create_render_pass(&info, None)? })
}

struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,

    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,

    descriptor_sets: Vec<vk::DescriptorSet>,

    uniform_buffer: Buffer,
    vertex_buffer: Buffer,

    mesh: Mesh,
}

impl Pipeline {
    fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        camera: &Camera,
    ) -> Result<Self> {
        let main = CString::new("main").unwrap();

        let vert_module = create_shader_module(
            &device,
            include_bytes_aligned_as!(u32, "../shaders/vert.spv"),
        )?;

        let frag_module = create_shader_module(
            &device,
            include_bytes_aligned_as!(u32, "../shaders/frag.spv"),
        )?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&main);

        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&main);

        let shader_stages = [vert_stage.build(), frag_stage.build()];

        let vert_attrib_desc = [vk::VertexInputAttributeDescription {
            format: vk::Format::R32G32B32_SFLOAT,
            binding: 0,
            location: 0,
            offset: 0,
        }];

        let vert_binding_desc = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];

        let vert_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vert_attrib_desc)
            .vertex_binding_descriptions(&vert_binding_desc);

        let vert_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let mesh = Mesh::from_obj(Path::new("models/cat/cat.obj"))?;

        let vertex_buffer = {
            let create_info = vk::BufferCreateInfo::builder()
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .size(mesh.size());

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let mut buffer = Buffer::new(device, &create_info, memory_flags, memory_props)?;

            buffer.fill(device, &mesh.verts)?;
            buffer.unmap(device);

            buffer
        };

        let descriptor_pool = unsafe {
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain.image_count,
            }];

            let info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&sizes)
                .max_sets(swapchain.image_count);

            device.create_descriptor_pool(&info, None)?
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

            device.create_descriptor_set_layout(&layout_info, None)?
        };

        let uniform_buffer = {
            let create_info = vk::BufferCreateInfo::builder()
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .size(128);

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let mut buffer = Buffer::new(device, &create_info, memory_flags, memory_props)?;

            buffer.map(device)?;

            upload_camera_to_uniform_buffer(camera, &buffer);

            buffer
        };

        let descriptor_sets = unsafe {
            let layouts = vec![descriptor_layout; swapchain.image_count as usize];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);

            device.allocate_descriptor_sets(&alloc_info)?
        };

        for set in descriptor_sets.iter() {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: 128,
            }];
            let writes = [vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { device.update_descriptor_sets(&writes, &[]) }
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

            device.create_pipeline_layout(&info, None)?
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
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            *device
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
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(Self {
            pipeline,
            layout,
            descriptor_pool,
            descriptor_layout,
            descriptor_sets,
            uniform_buffer,
            vertex_buffer,
            mesh,
        })
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);

            self.vertex_buffer.destroy(device);
            self.uniform_buffer.destroy(device);
        }
    }
}

fn upload_camera_to_uniform_buffer(camera: &Camera, buffer: &Buffer) {
    let slice: &mut [f32] = buffer.mapped_slice().unwrap();

    camera.perspective.write_cols_to_slice(slice);
    camera.view.write_cols_to_slice(&mut slice[16..]);
}

struct Queue {
    queue: vk::Queue,
    index: u32,
}

impl Queue {
    fn new(device: &ash::Device, index: u32) -> Self {
        Self {
            queue: unsafe { device.get_device_queue(index, 0) },
            index,
        }
    }
}

struct FrameSync {
    /// This get's signaled when the frame has been presented and is then available to draw to
    /// again.
    presented: vk::Semaphore,
    /// This get's signaled when the GPU is done drawing to the frame and the frame is then ready
    /// to be presented.
    rendered: vk::Semaphore,
    /// This is essentially the same as `presented`, but used to sync with the CPU.
    ready_to_draw: vk::Fence,
}

impl FrameSync {
    fn new(device: &ash::Device) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let presented = unsafe { device.create_semaphore(&semaphore_info, None)? };

        let rendered = unsafe { device.create_semaphore(&semaphore_info, None)? };

        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let ready_to_draw = unsafe { device.create_fence(&fence_info, None)? };

        Ok(Self {
            presented,
            rendered,
            ready_to_draw,
        })
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_semaphore(self.presented, None);
            device.destroy_semaphore(self.rendered, None);
            device.destroy_fence(self.ready_to_draw, None);
        }
    }
}

struct SwapchainImage {
    image: vk::Image,
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
}

impl SwapchainImage {
    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.view, None);
        }
    }
}

#[derive(Clone)]
struct Swapchain {
    loader: khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    image_count: u32,
}

impl Swapchain {
    /// Create a new swapchain. `extent` is used to determine the size of the swapchain images only
    /// if it aren't able to determine it from `surface`.
    ///
    /// In the case that `old` is some, it will use `old` to create the new swapchain, but not use
    /// it's format or extent. `old` will be destroyed and left in an invalid state.
    fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: &Queue,
        surface: &WaylandSurface,
        extent: vk::Extent2D,
        old: Option<Swapchain>,
    ) -> Result<Self> {
        let surface_formats = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(physical, surface.surface)?
        };
        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(physical, surface.surface)?
        };
        let surface_caps = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(physical, surface.surface)?
        };

        let queue_families = [graphics_queue.index];
        let min_image_count = 2.max(surface_caps.min_image_count);

        let format = surface_formats
            .iter()
            .find(|format| {
                format.format == ash::vk::Format::B8G8R8A8_SRGB
                    && format.color_space == ash::vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| surface_formats.first())
            .ok_or_else(|| anyhow!("can't find valid surface format"))?;

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
            .old_swapchain(
                old.as_ref()
                    .map(|sc| sc.swapchain)
                    .unwrap_or(vk::SwapchainKHR::null()),
            )
            .surface(surface.surface)
            .min_image_count(min_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode);

        let loader = old
            .as_ref()
            .map(|sc| sc.loader.clone())
            .unwrap_or_else(|| khr::Swapchain::new(instance, device));

        let swapchain = unsafe { loader.create_swapchain(&swapchain_info, None)? };

        if let Some(old) = old {
            old.destroy();
        }

        Ok(Self {
            swapchain,
            loader,
            extent,
            format: format.format,
            image_count: min_image_count,
        })
    }

    fn create_images(
        &mut self,
        device: &ash::Device,
        render_pass: vk::RenderPass,
    ) -> Result<Vec<SwapchainImage>> {
        let images = unsafe { self.loader.get_swapchain_images(self.swapchain)? };

        self.image_count = images.len() as u32;

        trace!("using {} swap chain images", images.len());

        let swap_chain_images: Result<Vec<_>> = images
            .iter()
            .map(|image| {
                let view = unsafe {
                    let subresource_range = vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1);
                    let view_info = vk::ImageViewCreateInfo::builder()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format)
                        .subresource_range(*subresource_range);

                    device.create_image_view(&view_info, None)?
                };

                let framebuffer = unsafe {
                    let views = [view];
                    let info = vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass)
                        .attachments(&views)
                        .width(self.extent.width)
                        .height(self.extent.height)
                        .layers(1);

                    device.create_framebuffer(&info, None)?
                };
                Ok(SwapchainImage {
                    image: *image,
                    view,
                    framebuffer,
                })
            })
            .collect();

        Ok(swap_chain_images?)
    }

    fn destroy(self) {
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
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

/// Have two frames in flight at a time. You could easily have more, but you risk having the CPU
/// run multiple frames ahead, which causes latency.
const FRAMES_IN_FLIGHT: usize = 2;
