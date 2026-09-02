use anyhow::{bail, Context, Result};
use ash::{vk, Entry};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum ExecutionMode { Normal, Xtx7900 }

#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct XtxTuning {
    pub gpu_profile: bool,
    pub cooperative_matrix: bool,
}

const GPU_QUERY_CAPACITY: u32 = 16_384;

impl ExecutionMode {
    pub fn as_str(self) -> &'static str { match self { Self::Normal => "normal", Self::Xtx7900 => "xtx7900" } }
}

use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    io::Cursor,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub api_version: String,
    pub driver_version: u32,
    pub compute_queue_family: Option<u32>,
    pub shader_float16: bool,
    pub storage_buffer_16bit: bool,
    pub subgroup_size: u32,
    pub subgroup_arithmetic: bool,
    pub subgroup_size_control: bool,
    pub compute_full_subgroups: bool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub required_subgroup_size_compute: bool,
    pub cooperative_matrix: bool,
    pub cooperative_matrix_16x16x16_f16_f32: bool,
    pub local_heap_bytes: u64,
    pub max_storage_buffer_range: u64,
    pub max_compute_work_group_count_x: u32,
    pub timestamp_valid_bits: u32,
    pub timestamp_period_ns: f32,
}


#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuTimingStat {
    pub total_ms: f64,
    pub calls: u64,
    pub avg_ms: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuProfileSnapshot {
    pub enabled: bool,
    pub timings: BTreeMap<String, GpuTimingStat>,
}

#[derive(Clone)]
struct PendingGpuSpan { name: &'static str, start: u32, end: u32 }

#[derive(Default)]
struct GpuProfileState {
    next_query: u32,
    pending: Vec<PendingGpuSpan>,
    totals: BTreeMap<String, (f64, u64)>,
}

pub struct GpuBuffer {
    device: Arc<ash::Device>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub memory_properties: vk::MemoryPropertyFlags,
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: Arc<ash::Device>,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    pub pipeline_cache: vk::PipelineCache,
    pub timestamp_pool: vk::QueryPool,
    pub info: DeviceInfo,
    pub mode: ExecutionMode,
    pub xtx_tuning: XtxTuning,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    submit_fence: Mutex<vk::Fence>,
    gpu_profile: Mutex<GpuProfileState>,
}

fn version_text(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}
fn device_type_text(v: vk::PhysicalDeviceType) -> String {
    format!("{v:?}")
}

fn make_instance(entry: &Entry) -> Result<ash::Instance> {
    let app = CString::new("VoxGen")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app)
        .application_version(vk::make_api_version(0, 0, 6, 0))
        .engine_name(&app)
        .engine_version(vk::make_api_version(0, 0, 6, 0))
        .api_version(vk::API_VERSION_1_2);
    let ci = vk::InstanceCreateInfo::default().application_info(&app_info);
    Ok(unsafe { entry.create_instance(&ci, None) }.context("vkCreateInstance")?)
}

fn inspect(entry: &Entry, instance: &ash::Instance, physical: vk::PhysicalDevice, index: usize) -> DeviceInfo {
    let props = unsafe { instance.get_physical_device_properties(physical) };
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let compute_queue_family = queues.iter().enumerate().find_map(|(i, q)| {
        q.queue_flags
            .contains(vk::QueueFlags::COMPUTE)
            .then_some(i as u32)
    });
    let timestamp_valid_bits = compute_queue_family
        .and_then(|i| queues.get(i as usize))
        .map(|q| q.timestamp_valid_bits)
        .unwrap_or(0);

    let mem = unsafe { instance.get_physical_device_memory_properties(physical) };
    let mut local_heap_bytes = 0u64;
    for i in 0..mem.memory_heap_count as usize {
        let h = mem.memory_heaps[i];
        if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            local_heap_bytes = local_heap_bytes.saturating_add(h.size);
        }
    }

    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
        .unwrap_or_default();
    let has_extension = |wanted: &CStr| {
        extensions.iter().any(|ext| {
            ext.extension_name_as_c_str()
                .map(|name| name == wanted)
                .unwrap_or(false)
        })
    };
    let has_subgroup_size_control = has_extension(ash::ext::subgroup_size_control::NAME);
    let has_cooperative_matrix = has_extension(ash::khr::cooperative_matrix::NAME);

    let mut f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    let mut feats2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f16)
        .push_next(&mut storage16);
    unsafe {
        instance.get_physical_device_properties2(physical, &mut props2);
        instance.get_physical_device_features2(physical, &mut feats2);
    }

    let mut subgroup_size_control = false;
    let mut compute_full_subgroups = false;
    let mut min_subgroup_size = subgroup.subgroup_size;
    let mut max_subgroup_size = subgroup.subgroup_size;
    let mut required_subgroup_size_compute = false;
    if has_subgroup_size_control {
        let mut sc_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        let mut sc_properties = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut sc_features);
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sc_properties);
        unsafe {
            instance.get_physical_device_features2(physical, &mut f2);
            instance.get_physical_device_properties2(physical, &mut p2);
        }
        subgroup_size_control = sc_features.subgroup_size_control == vk::TRUE;
        compute_full_subgroups = sc_features.compute_full_subgroups == vk::TRUE;
        min_subgroup_size = sc_properties.min_subgroup_size;
        max_subgroup_size = sc_properties.max_subgroup_size;
        required_subgroup_size_compute = sc_properties
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE);
    }

    let mut cooperative_matrix = false;
    let mut cooperative_matrix_16x16x16_f16_f32 = false;
    if has_cooperative_matrix {
        let mut cm_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut cm_features);
        unsafe { instance.get_physical_device_features2(physical, &mut f2); }
        cooperative_matrix = cm_features.cooperative_matrix == vk::TRUE;
        if cooperative_matrix {
            let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
            if let Ok(properties) = unsafe {
                loader.get_physical_device_cooperative_matrix_properties(physical)
            } {
                cooperative_matrix_16x16x16_f16_f32 = properties.iter().any(|p| {
                    p.m_size == 16
                        && p.n_size == 16
                        && p.k_size == 16
                        && p.a_type == vk::ComponentTypeKHR::FLOAT16
                        && p.b_type == vk::ComponentTypeKHR::FLOAT16
                        && p.c_type == vk::ComponentTypeKHR::FLOAT32
                        && p.result_type == vk::ComponentTypeKHR::FLOAT32
                        && p.scope == vk::ScopeKHR::SUBGROUP
                        && p.saturating_accumulation == vk::FALSE
                });
            }
        }
    }

    DeviceInfo {
        index,
        name,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        device_type: device_type_text(props.device_type),
        api_version: version_text(props.api_version),
        driver_version: props.driver_version,
        compute_queue_family,
        shader_float16: f16.shader_float16 == vk::TRUE,
        storage_buffer_16bit: storage16.storage_buffer16_bit_access == vk::TRUE,
        subgroup_size: subgroup.subgroup_size,
        subgroup_arithmetic: subgroup.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
            && subgroup.supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC),
        subgroup_size_control,
        compute_full_subgroups,
        min_subgroup_size,
        max_subgroup_size,
        required_subgroup_size_compute,
        cooperative_matrix,
        cooperative_matrix_16x16x16_f16_f32,
        local_heap_bytes,
        max_storage_buffer_range: props.limits.max_storage_buffer_range as u64,
        max_compute_work_group_count_x: props.limits.max_compute_work_group_count[0],
        timestamp_valid_bits,
        timestamp_period_ns: props.limits.timestamp_period,
    }
}

pub fn enumerate_devices() -> Result<Vec<DeviceInfo>> {
    let entry = unsafe { Entry::load() }.context("load Vulkan loader")?;
    let instance = make_instance(&entry)?;
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .context("vkEnumeratePhysicalDevices")?;
    let out = physicals
        .iter()
        .enumerate()
        .map(|(i, p)| inspect(&entry, &instance, *p, i))
        .collect();
    unsafe {
        instance.destroy_instance(None);
    }
    Ok(out)
}

impl VulkanContext {
    pub fn new(requested_index: Option<usize>, mode: ExecutionMode, xtx_tuning: XtxTuning) -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("load Vulkan loader")?;
        let instance = make_instance(&entry)?;
        let physicals = unsafe { instance.enumerate_physical_devices() }
            .context("vkEnumeratePhysicalDevices")?;
        if physicals.is_empty() {
            unsafe { instance.destroy_instance(None) };
            bail!("No Vulkan devices found. CPU fallback is deliberately disabled.");
        }
        let infos: Vec<_> = physicals
            .iter()
            .enumerate()
            .map(|(i, p)| inspect(&entry, &instance, *p, i))
            .collect();
        let selected = if let Some(i) = requested_index {
            infos
                .get(i)
                .filter(|x| x.compute_queue_family.is_some())
                .map(|_| i)
                .with_context(|| format!("Vulkan device {i} is unavailable or has no compute queue"))?
        } else if mode == ExecutionMode::Xtx7900 {
            infos
                .iter()
                .enumerate()
                .find(|(_, x)| {
                    x.compute_queue_family.is_some()
                        && x.vendor_id == 0x1002
                        && x.name.to_ascii_lowercase().contains("7900 xtx")
                })
                .map(|(i, _)| i)
                .context("--mode xtx7900 requested, but no compute-capable AMD Radeon RX 7900 XTX was found")?
        } else {
            infos
                .iter()
                .enumerate()
                .filter(|(_, x)| x.compute_queue_family.is_some())
                .max_by_key(|(_, x)| {
                    let discrete = x.device_type.contains("DISCRETE") as u64;
                    let amd = (x.vendor_id == 0x1002) as u64;
                    (discrete << 62)
                        | (amd << 61)
                        | x.local_heap_bytes.min((1u64 << 61) - 1)
                })
                .map(|(i, _)| i)
                .context("No Vulkan compute-capable device found")?
        };
        let info = infos[selected].clone();
        if mode == ExecutionMode::Xtx7900 {
            let name = info.name.to_ascii_lowercase();
            if info.vendor_id != 0x1002 || !name.contains("7900 xtx") {
                unsafe { instance.destroy_instance(None) };
                bail!(
                    "--mode xtx7900 requires an AMD Radeon RX 7900 XTX; selected Vulkan{} is '{}' (vendor=0x{:04x}, device=0x{:04x}). Use --mode normal instead.",
                    info.index, info.name, info.vendor_id, info.device_id
                );
            }
            if !info.subgroup_arithmetic || !matches!(info.subgroup_size, 32 | 64) {
                unsafe { instance.destroy_instance(None) };
                bail!(
                    "--mode xtx7900 requires compute subgroup arithmetic with subgroup size 32 or 64; '{}' reports arithmetic={} subgroup={}",
                    info.name, info.subgroup_arithmetic, info.subgroup_size
                );
            }
            if !info.subgroup_size_control
                || !info.required_subgroup_size_compute
                || info.min_subgroup_size > 32
                || info.max_subgroup_size < 32
            {
                unsafe { instance.destroy_instance(None) };
                bail!(
                    "--mode xtx7900 requires VK_EXT_subgroup_size_control with compute wave32 support; '{}' reports control={} compute-stage={} range={}..{}",
                    info.name, info.subgroup_size_control, info.required_subgroup_size_compute,
                    info.min_subgroup_size, info.max_subgroup_size
                );
            }
        }
        let physical = physicals[selected];
        let queue_family = info.compute_queue_family.unwrap();
        let priorities = [1.0f32];
        let queue_ci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];

        // BaseLM kernels intentionally use packed u32 storage and unpackHalf2x16,
        // so native 16-bit storage is not required for BaseLM/ResidualLM kernels. Enable it when
        // the device exposes it because later acoustic kernels can use it directly.
        let mut f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(info.shader_float16);
        let mut storage16 = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(info.storage_buffer_16bit);
        let mut subgroup_control = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
            .subgroup_size_control(mode == ExecutionMode::Xtx7900)
            .compute_full_subgroups(mode == ExecutionMode::Xtx7900 && info.compute_full_subgroups);
        let enable_coopmat = mode == ExecutionMode::Xtx7900
            && xtx_tuning.cooperative_matrix
            && info.cooperative_matrix_16x16x16_f16_f32
            && info.shader_float16;
        let mut coopmat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(enable_coopmat);
        let mut extension_names = Vec::new();
        if mode == ExecutionMode::Xtx7900 {
            extension_names.push(ash::ext::subgroup_size_control::NAME.as_ptr());
            if enable_coopmat {
                extension_names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
            }
        }
        let mut device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_ci)
            .enabled_extension_names(&extension_names)
            .push_next(&mut f16)
            .push_next(&mut storage16);
        if mode == ExecutionMode::Xtx7900 {
            device_ci = device_ci.push_next(&mut subgroup_control);
            if enable_coopmat {
                device_ci = device_ci.push_next(&mut coopmat);
            }
        }
        let device = Arc::new(
            unsafe { instance.create_device(physical, &device_ci, None) }
                .context("vkCreateDevice")?,
        );
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .context("create persistent command pool")?;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 16384,
        }];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(4096)
                    .pool_sizes(&pool_sizes)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                None,
            )
        }
        .context("create persistent descriptor pool")?;
        let pipeline_cache = unsafe {
            device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        }
        .context("create pipeline cache")?;
        let timestamp_pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(GPU_QUERY_CAPACITY),
                None,
            )
        }
        .context("create timestamp query pool")?;
        let submit_fence = unsafe {
            device.create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("create persistent submit fence")?;
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };

        Ok(Self {
            entry,
            instance,
            physical,
            device,
            queue_family,
            queue,
            command_pool,
            descriptor_pool,
            pipeline_cache,
            timestamp_pool,
            info,
            mode,
            xtx_tuning,
            memory_properties,
            submit_fence: Mutex::new(submit_fence),
            gpu_profile: Mutex::new(GpuProfileState::default()),
        })
    }


    pub fn select_spirv<'a>(&self, normal: &'a [u8], xtx7900: &'a [u8]) -> &'a [u8] {
        match self.mode { ExecutionMode::Normal => normal, ExecutionMode::Xtx7900 => xtx7900 }
    }

    fn memory_type_index(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            let bit = 1u32 << i;
            let flags = self.memory_properties.memory_types[i as usize].property_flags;
            if type_bits & bit != 0 && flags.contains(required) {
                return Ok(i);
            }
        }
        bail!("No Vulkan memory type satisfies {required:?} (type bits 0x{type_bits:08x})")
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<GpuBuffer> {
        let size = size.max(4);
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .context("vkCreateBuffer")?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = self.memory_type_index(req.memory_type_bits, properties)?;
        let memory = match unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type_index),
                None,
            )
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(e).context("vkAllocateMemory");
            }
        };
        if let Err(e) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            return Err(e).context("vkBindBufferMemory");
        }
        Ok(GpuBuffer {
            device: self.device.clone(),
            buffer,
            memory,
            size,
            memory_properties: properties,
        })
    }

    pub fn allocate_primary_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let buffers = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .context("allocate command buffer")?;
        Ok(buffers[0])
    }

    pub fn begin_one_time(&self, cmd: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .context("reset command buffer")?;
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .context("begin command buffer")?;
        }
        if self.gpu_profiling_enabled() {
            if let Ok(mut profile) = self.gpu_profile.lock() {
                profile.next_query = 0;
                profile.pending.clear();
            }
            unsafe { self.device.cmd_reset_query_pool(cmd, self.timestamp_pool, 0, GPU_QUERY_CAPACITY); }
        }
        Ok(())
    }

    pub fn gpu_profiling_enabled(&self) -> bool {
        self.mode == ExecutionMode::Xtx7900
            && self.xtx_tuning.gpu_profile
            && self.info.timestamp_valid_bits > 0
            && self.info.timestamp_period_ns > 0.0
    }

    pub fn xtx_coopmat_enabled(&self) -> bool {
        self.mode == ExecutionMode::Xtx7900
            && self.xtx_tuning.cooperative_matrix
            && self.info.cooperative_matrix_16x16x16_f16_f32
            && self.info.shader_float16
    }

    pub fn gpu_profile_begin(&self, cmd: vk::CommandBuffer, name: &'static str) -> Option<u32> {
        if !self.gpu_profiling_enabled() { return None; }
        let mut state = self.gpu_profile.lock().ok()?;
        if state.next_query + 2 > GPU_QUERY_CAPACITY { return None; }
        let start = state.next_query;
        let end = start + 1;
        state.next_query += 2;
        state.pending.push(PendingGpuSpan { name, start, end });
        drop(state);
        unsafe {
            self.device.cmd_write_timestamp(
                cmd, vk::PipelineStageFlags::COMPUTE_SHADER, self.timestamp_pool, start
            );
        }
        Some(end)
    }

    pub fn gpu_profile_end(&self, cmd: vk::CommandBuffer, end_query: Option<u32>) {
        if let Some(query) = end_query {
            unsafe {
                self.device.cmd_write_timestamp(
                    cmd, vk::PipelineStageFlags::COMPUTE_SHADER, self.timestamp_pool, query
                );
            }
        }
    }

    pub fn reset_gpu_profile(&self) {
        if let Ok(mut state) = self.gpu_profile.lock() {
            state.next_query = 0;
            state.pending.clear();
            state.totals.clear();
        }
    }

    pub fn gpu_profile_snapshot(&self) -> GpuProfileSnapshot {
        if !self.gpu_profiling_enabled() {
            return GpuProfileSnapshot { enabled: false, timings: BTreeMap::new() };
        }
        let Ok(state) = self.gpu_profile.lock() else {
            return GpuProfileSnapshot { enabled: true, timings: BTreeMap::new() };
        };
        let timings = state.totals.iter().map(|(name, &(total_ms, calls))| {
            (name.clone(), GpuTimingStat {
                total_ms, calls, avg_ms: if calls > 0 { total_ms / calls as f64 } else { 0.0 },
            })
        }).collect();
        GpuProfileSnapshot { enabled: true, timings }
    }

    pub fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<()> {
        // VoxGen currently exposes one Vulkan queue. Reusing a single fence avoids a
        // vkCreateFence/vkDestroyFence pair for every tiny inference submission, while
        // this mutex also provides the host-side external synchronization required for
        // submissions to the same VkQueue.
        let fence_guard = self
            .submit_fence
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan submit fence lock is poisoned"))?;
        let fence = *fence_guard;
        unsafe {
            self.device.end_command_buffer(cmd).context("end command buffer")?;
            let cbs = [cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&cbs)];
            self.device.reset_fences(&[fence]).context("reset Vulkan submit fence")?;
            self.device.queue_submit(self.queue, &submits, fence).context("queue submit")?;
            self.device.wait_for_fences(&[fence], true, u64::MAX).context("wait for Vulkan fence")?;
        }
        self.collect_gpu_profile();
        Ok(())
    }

    fn collect_gpu_profile(&self) {
        if !self.gpu_profiling_enabled() { return; }
        let (query_count, pending) = match self.gpu_profile.lock() {
            Ok(state) => (state.next_query, state.pending.clone()),
            Err(_) => return,
        };
        if query_count == 0 || pending.is_empty() { return; }
        let mut values = vec![0u64; query_count as usize];
        let result = unsafe {
            self.device.get_query_pool_results(
                self.timestamp_pool, 0, &mut values,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if result.is_err() { return; }
        let valid_bits = self.info.timestamp_valid_bits;
        let mask = if valid_bits >= 64 { u64::MAX } else { (1u64 << valid_bits) - 1 };
        if let Ok(mut state) = self.gpu_profile.lock() {
            for span in pending {
                let a = values[span.start as usize] & mask;
                let b = values[span.end as usize] & mask;
                let ticks = if valid_bits >= 64 { b.wrapping_sub(a) } else { b.wrapping_sub(a) & mask };
                let ms = ticks as f64 * self.info.timestamp_period_ns as f64 / 1_000_000.0;
                if ms.is_finite() {
                    let entry = state.totals.entry(span.name.to_string()).or_insert((0.0, 0));
                    entry.0 += ms;
                    entry.1 += 1;
                }
            }
        }
    }

    pub fn compute_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    /// Narrow compute dependency for explicitly shared buffers. This keeps unrelated
    /// storage buffers out of the cache/memory dependency while preserving the same
    /// shader-write -> shader-read/write ordering as `compute_barrier`.
    pub fn compute_buffer_barrier(&self, cmd: vk::CommandBuffer, buffers: &[&GpuBuffer]) {
        if buffers.is_empty() { return; }
        let barriers: Vec<_> = buffers.iter().map(|b| {
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(b.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)
        }).collect();
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
    }

    pub fn transfer_to_compute_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    pub fn compute_to_transfer_rw_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    pub fn compute_to_transfer_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    fn shader_or_transfer_to_transfer_read_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    pub fn upload_device_local(&self, bytes: &[u8]) -> Result<GpuBuffer> {
        let device_buf = self.create_buffer(
            bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        if bytes.is_empty() {
            return Ok(device_buf);
        }

        let staging_size = (256usize * 1024 * 1024).min(bytes.len()).max(4);
        let staging = self.create_buffer(
            staging_size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let cmd = self.allocate_primary_command_buffer()?;
        let mut offset = 0usize;
        while offset < bytes.len() {
            let n = staging_size.min(bytes.len() - offset);
            unsafe {
                let ptr = self
                    .device
                    .map_memory(staging.memory, 0, n as u64, vk::MemoryMapFlags::empty())
                    .context("map staging memory")? as *mut u8;
                std::ptr::copy_nonoverlapping(bytes[offset..offset + n].as_ptr(), ptr, n);
                self.device.unmap_memory(staging.memory);
            }
            self.begin_one_time(cmd)?;
            let copy = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: offset as u64,
                size: n as u64,
            }];
            unsafe {
                self.device
                    .cmd_copy_buffer(cmd, staging.buffer, device_buf.buffer, &copy);
            }
            self.transfer_to_compute_barrier(cmd);
            self.submit_and_wait(cmd)?;
            offset += n;
        }
        unsafe {
            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }
        Ok(device_buf)
    }

    pub fn upload_f32(&self, dst: &GpuBuffer, values: &[f32]) -> Result<()> {
        let cmd = self.allocate_primary_command_buffer()?;
        self.begin_one_time(cmd)?;
        let staging = self.record_upload_f32(cmd, dst, values)?;
        self.submit_and_wait(cmd)?;
        unsafe { self.device.free_command_buffers(self.command_pool, &[cmd]); }
        drop(staging);
        Ok(())
    }

    /// Record a host->device f32 upload into an already-open command buffer.
    /// The returned staging buffer must stay alive until that command buffer has completed.
    pub fn record_upload_f32(&self, cmd: vk::CommandBuffer, dst: &GpuBuffer, values: &[f32]) -> Result<GpuBuffer> {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        };
        if bytes.is_empty() {
            bail!("recorded f32 upload cannot be empty");
        }
        if bytes.len() as u64 > dst.size {
            bail!("upload of {} bytes exceeds destination buffer of {} bytes", bytes.len(), dst.size);
        }
        let staging = self.create_buffer(
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = self
                .device
                .map_memory(staging.memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())
                .context("map upload memory")? as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            self.device.unmap_memory(staging.memory);
        }
        unsafe {
            self.device.cmd_copy_buffer(
                cmd,
                staging.buffer,
                dst.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes.len() as u64,
                }],
            );
        }
        self.transfer_to_compute_barrier(cmd);
        Ok(staging)
    }

    pub fn read_f32(&self, src: &GpuBuffer, count: usize) -> Result<Vec<f32>> {
        let cmd = self.allocate_primary_command_buffer()?;
        self.begin_one_time(cmd)?;
        let out = self.submit_and_read_f32(cmd, src, count)?;
        unsafe { self.device.free_command_buffers(self.command_pool, &[cmd]); }
        Ok(out)
    }

    /// Append a device->host f32 readback to an already-open command buffer, submit it,
    /// wait once, then map the staging allocation. This is the key primitive used to
    /// fuse inference work and its tiny result readbacks into one queue submission.
    pub fn submit_and_read_f32(&self, cmd: vk::CommandBuffer, src: &GpuBuffer, count: usize) -> Result<Vec<f32>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .context("readback size overflow")?;
        if bytes == 0 {
            bail!("recorded f32 readback cannot be empty");
        }
        if bytes as u64 > src.size {
            bail!("readback of {bytes} bytes exceeds source buffer of {} bytes", src.size);
        }
        let staging = self.create_buffer(
            bytes as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.shader_or_transfer_to_transfer_read_barrier(cmd);
        unsafe {
            self.device.cmd_copy_buffer(
                cmd,
                src.buffer,
                staging.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes as u64,
                }],
            );
        }
        self.submit_and_wait(cmd)?;
        let mut out = vec![0f32; count];
        unsafe {
            let ptr = self
                .device
                .map_memory(staging.memory, 0, bytes as u64, vk::MemoryMapFlags::empty())
                .context("map readback memory")? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr() as *mut u8, bytes);
            self.device.unmap_memory(staging.memory);
        }
        Ok(out)
    }

    pub fn create_compute_pipeline(
        &self,
        spirv: &[u8],
        binding_count: u32,
        push_constant_bytes: u32,
    ) -> Result<ComputePipeline> {
        let mut cursor = Cursor::new(spirv);
        let words = ash::util::read_spv(&mut cursor).context("read embedded SPIR-V")?;
        let shader = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&words),
                None,
            )
        }
        .context("create shader module")?;

        let bindings: Vec<_> = (0..binding_count)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = match unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        } {
            Ok(v) => v,
            Err(e) => {
                unsafe { self.device.destroy_shader_module(shader, None) };
                return Err(e).context("create descriptor set layout");
            }
        };

        let set_layouts = [set_layout];
        let ranges = if push_constant_bytes > 0 {
            vec![vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constant_bytes)]
        } else {
            Vec::new()
        };
        let layout = match unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        } {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    self.device.destroy_descriptor_set_layout(set_layout, None);
                    self.device.destroy_shader_module(shader, None);
                }
                return Err(e).context("create pipeline layout");
            }
        };

        let entry = CString::new("main")?;
        let mut required_subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(32);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry);
        if self.mode == ExecutionMode::Xtx7900 {
            stage = stage.push_next(&mut required_subgroup);
        }
        let ci = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline = match unsafe {
            self.device
                .create_compute_pipelines(self.pipeline_cache, &ci, None)
        } {
            Ok(v) => v[0],
            Err((_, e)) => {
                unsafe {
                    self.device.destroy_pipeline_layout(layout, None);
                    self.device.destroy_descriptor_set_layout(set_layout, None);
                    self.device.destroy_shader_module(shader, None);
                }
                return Err(e).context("create compute pipeline");
            }
        };
        unsafe { self.device.destroy_shader_module(shader, None); }

        let set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .context("allocate descriptor set")?[0];

        Ok(ComputePipeline {
            device: self.device.clone(),
            descriptor_pool: self.descriptor_pool,
            pipeline,
            layout,
            set_layout,
            set,
        })
    }
}

pub struct ComputePipeline {
    device: Arc<ash::Device>,
    descriptor_pool: vk::DescriptorPool,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
}

impl ComputePipeline {
    pub fn bind_buffers(&self, buffers: &[&GpuBuffer]) {
        let infos: Vec<_> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(b.size)
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]); }
    }

    pub fn bind(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
        }
    }

    pub fn push<T: bytemuck::Pod>(&self, cmd: vk::CommandBuffer, value: &T) {
        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(value),
            );
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &[self.set]);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(fence) = self.submit_fence.get_mut() {
                self.device.destroy_fence(*fence, None);
            }
            self.device.destroy_query_pool(self.timestamp_pool, None);
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
