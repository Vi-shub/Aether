//! High-level buffer abstractions.
//!
//! Provides typed buffers that integrate with Aether's state system.

use std::sync::Arc;
use aether_core::{StateId, DType};

use crate::device::CudaDevice;
use crate::memory::{DeviceMemory, HostMemory, PinnedMemory};
use crate::error::{CudaError, Result};

// =============================================================================
// GPU Buffer
// =============================================================================

/// GPU buffer with metadata.
///
/// This is the primary interface for GPU memory in Aether.
/// It tracks the state ID, dtype, shape, and underlying memory.
#[derive(Debug)]
pub struct GpuBuffer {
    /// Associated state ID (if any).
    state_id: Option<StateId>,
    /// Data type.
    dtype: DType,
    /// Shape (dimensions).
    shape: Vec<usize>,
    /// Number of elements.
    numel: usize,
    /// Underlying device memory.
    memory: DeviceMemory,
}

impl GpuBuffer {
    /// Allocate a new GPU buffer.
    pub fn allocate(
        device: Arc<CudaDevice>,
        dtype: DType,
        shape: &[usize],
    ) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let size_bytes = (numel * dtype.size_bytes()) as u64;

        let memory = DeviceMemory::allocate(device, size_bytes)?;

        Ok(Self {
            state_id: None,
            dtype,
            shape: shape.to_vec(),
            numel,
            memory,
        })
    }

    /// Allocate with a state ID.
    pub fn allocate_for_state(
        device: Arc<CudaDevice>,
        state_id: StateId,
        dtype: DType,
        shape: &[usize],
    ) -> Result<Self> {
        let mut buffer = Self::allocate(device, dtype, shape)?;
        buffer.state_id = Some(state_id);
        Ok(buffer)
    }

    /// Get state ID.
    pub fn state_id(&self) -> Option<&StateId> {
        self.state_id.as_ref()
    }

    /// Set state ID.
    pub fn set_state_id(&mut self, id: StateId) {
        self.state_id = Some(id);
    }

    /// Get dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.memory.size()
    }

    /// Get underlying memory.
    pub fn memory(&self) -> &DeviceMemory {
        &self.memory
    }

    /// Get mutable underlying memory.
    pub fn memory_mut(&mut self) -> &mut DeviceMemory {
        &mut self.memory
    }

    /// Get device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        self.memory.device()
    }

    /// Reshape buffer (must have same total elements).
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<()> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel {
            return Err(CudaError::InvalidBuffer(format!(
                "Cannot reshape {} elements to {} elements",
                self.numel, new_numel
            )));
        }
        self.shape = new_shape.to_vec();
        Ok(())
    }

    /// View with different shape (no copy).
    pub fn view(&self, new_shape: &[usize]) -> Result<GpuBufferView<'_>> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel {
            return Err(CudaError::InvalidBuffer(format!(
                "Cannot view {} elements as {} elements",
                self.numel, new_numel
            )));
        }
        Ok(GpuBufferView {
            buffer: self,
            shape: new_shape.to_vec(),
        })
    }
}

/// View into a GPU buffer with different shape.
#[derive(Debug)]
pub struct GpuBufferView<'a> {
    buffer: &'a GpuBuffer,
    shape: Vec<usize>,
}

impl<'a> GpuBufferView<'a> {
    pub fn dtype(&self) -> DType {
        self.buffer.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.buffer.numel
    }

    pub fn memory(&self) -> &DeviceMemory {
        &self.buffer.memory
    }
}

// =============================================================================
// CPU Buffer
// =============================================================================

/// CPU buffer with metadata.
///
/// Mirrors GpuBuffer but for CPU memory.
#[derive(Debug)]
pub struct CpuBuffer {
    /// Associated state ID (if any).
    state_id: Option<StateId>,
    /// Data type.
    dtype: DType,
    /// Shape (dimensions).
    shape: Vec<usize>,
    /// Number of elements.
    numel: usize,
    /// Underlying host memory.
    memory: HostMemory,
    /// Whether this is pinned memory.
    is_pinned: bool,
}

impl CpuBuffer {
    /// Allocate a new CPU buffer.
    pub fn allocate(dtype: DType, shape: &[usize]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let size_bytes = (numel * dtype.size_bytes()) as u64;

        let memory = HostMemory::allocate(size_bytes)?;

        Ok(Self {
            state_id: None,
            dtype,
            shape: shape.to_vec(),
            numel,
            memory,
            is_pinned: false,
        })
    }

    /// Create from existing data.
    pub fn from_data(dtype: DType, shape: &[usize], data: Vec<u8>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.size_bytes();

        if data.len() != expected_bytes {
            return Err(CudaError::InvalidBuffer(format!(
                "Data size {} doesn't match expected {} bytes",
                data.len(),
                expected_bytes
            )));
        }

        Ok(Self {
            state_id: None,
            dtype,
            shape: shape.to_vec(),
            numel,
            memory: HostMemory::from_vec(data),
            is_pinned: false,
        })
    }

    /// Get state ID.
    pub fn state_id(&self) -> Option<&StateId> {
        self.state_id.as_ref()
    }

    /// Set state ID.
    pub fn set_state_id(&mut self, id: StateId) {
        self.state_id = Some(id);
    }

    /// Get dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.memory.size()
    }

    /// Check if pinned.
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Get data slice.
    pub fn as_slice(&self) -> &[u8] {
        self.memory.as_slice()
    }

    /// Get mutable data slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.memory.as_mut_slice()
    }

    /// Get underlying memory.
    pub fn memory(&self) -> &HostMemory {
        &self.memory
    }

    /// Reshape buffer.
    pub fn reshape(&mut self, new_shape: &[usize]) -> Result<()> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel {
            return Err(CudaError::InvalidBuffer(format!(
                "Cannot reshape {} elements to {} elements",
                self.numel, new_numel
            )));
        }
        self.shape = new_shape.to_vec();
        Ok(())
    }
}

// =============================================================================
// Buffer Registry
// =============================================================================

/// Registry for tracking GPU buffers by state ID.
pub struct BufferRegistry {
    /// GPU buffers indexed by state ID.
    gpu_buffers: parking_lot::RwLock<std::collections::HashMap<StateId, GpuBuffer>>,
    /// CPU buffers indexed by state ID.
    cpu_buffers: parking_lot::RwLock<std::collections::HashMap<StateId, CpuBuffer>>,
}

impl BufferRegistry {
    /// Create a new registry.
    pub fn new() -> Self {
        Self {
            gpu_buffers: parking_lot::RwLock::new(std::collections::HashMap::new()),
            cpu_buffers: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Register a GPU buffer.
    pub fn register_gpu(&self, buffer: GpuBuffer) -> Option<StateId> {
        let id = buffer.state_id.clone()?;
        self.gpu_buffers.write().insert(id.clone(), buffer);
        Some(id)
    }

    /// Register a CPU buffer.
    pub fn register_cpu(&self, buffer: CpuBuffer) -> Option<StateId> {
        let id = buffer.state_id.clone()?;
        self.cpu_buffers.write().insert(id.clone(), buffer);
        Some(id)
    }

    /// Get a GPU buffer.
    pub fn get_gpu(&self, id: &StateId) -> Option<GpuBuffer> {
        self.gpu_buffers.write().remove(id)
    }

    /// Get a CPU buffer.
    pub fn get_cpu(&self, id: &StateId) -> Option<CpuBuffer> {
        self.cpu_buffers.write().remove(id)
    }

    /// Check if GPU buffer exists.
    pub fn has_gpu(&self, id: &StateId) -> bool {
        self.gpu_buffers.read().contains_key(id)
    }

    /// Check if CPU buffer exists.
    pub fn has_cpu(&self, id: &StateId) -> bool {
        self.cpu_buffers.read().contains_key(id)
    }

    /// Remove a GPU buffer.
    pub fn remove_gpu(&self, id: &StateId) -> Option<GpuBuffer> {
        self.gpu_buffers.write().remove(id)
    }

    /// Remove a CPU buffer.
    pub fn remove_cpu(&self, id: &StateId) -> Option<CpuBuffer> {
        self.cpu_buffers.write().remove(id)
    }

    /// Get total GPU memory used.
    pub fn gpu_memory_used(&self) -> u64 {
        self.gpu_buffers
            .read()
            .values()
            .map(|b| b.size_bytes())
            .sum()
    }

    /// Get total CPU memory used.
    pub fn cpu_memory_used(&self) -> u64 {
        self.cpu_buffers
            .read()
            .values()
            .map(|b| b.size_bytes())
            .sum()
    }
}

impl Default for BufferRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let buffer = GpuBuffer::allocate(device, DType::Float16, &[1024, 768]).unwrap();

        assert_eq!(buffer.shape(), &[1024, 768]);
        assert_eq!(buffer.numel(), 1024 * 768);
        assert_eq!(buffer.size_bytes(), (1024 * 768 * 2) as u64); // FP16 = 2 bytes
    }

    #[test]
    fn test_cpu_buffer() {
        let buffer = CpuBuffer::allocate(DType::Float32, &[256, 256]).unwrap();

        assert_eq!(buffer.shape(), &[256, 256]);
        assert_eq!(buffer.numel(), 256 * 256);
        assert_eq!(buffer.size_bytes(), (256 * 256 * 4) as u64); // FP32 = 4 bytes
    }

    #[test]
    fn test_buffer_reshape() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let mut buffer = GpuBuffer::allocate(device, DType::Float16, &[1024, 768]).unwrap();

        buffer.reshape(&[768, 1024]).unwrap();
        assert_eq!(buffer.shape(), &[768, 1024]);

        // Invalid reshape should fail
        assert!(buffer.reshape(&[100, 100]).is_err());
    }

    #[test]
    fn test_buffer_registry() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let registry = BufferRegistry::new();

        let mut buffer = GpuBuffer::allocate(device, DType::Float16, &[1024]).unwrap();
        buffer.set_state_id(StateId::new("test_state"));

        let id = registry.register_gpu(buffer).unwrap();
        assert!(registry.has_gpu(&id));
        assert!(!registry.has_cpu(&id));

        let buffer = registry.remove_gpu(&id).unwrap();
        assert!(!registry.has_gpu(&id));
        assert_eq!(buffer.state_id(), Some(&StateId::new("test_state")));
    }
}
