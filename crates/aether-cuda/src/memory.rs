//! Memory allocation primitives.
//!
//! Provides abstractions for:
//! - DeviceMemory: GPU HBM allocation
//! - HostMemory: CPU pageable memory
//! - PinnedMemory: CPU pinned memory (for fast DMA)

use std::sync::Arc;
use parking_lot::Mutex;

use crate::device::CudaDevice;
use crate::error::{CudaError, Result};

// =============================================================================
// Device Memory (GPU HBM)
// =============================================================================

/// GPU device memory allocation.
///
/// This represents an allocation in GPU HBM. Memory is automatically
/// freed when dropped.
pub struct DeviceMemory {
    /// Device this memory belongs to.
    device: Arc<CudaDevice>,
    /// Size in bytes.
    size: u64,
    /// Raw pointer (for mock mode).
    #[cfg(not(feature = "cuda"))]
    _ptr: u64,
    /// cudarc allocation.
    #[cfg(feature = "cuda")]
    allocation: cudarc::driver::CudaSlice<u8>,
}

impl DeviceMemory {
    /// Allocate device memory.
    pub fn allocate(device: Arc<CudaDevice>, size: u64) -> Result<Self> {
        if !device.can_allocate(size) {
            return Err(CudaError::out_of_memory(size, device.free_memory()));
        }

        #[cfg(feature = "cuda")]
        {
            let allocation = device
                .raw_device()
                .alloc_zeros::<u8>(size as usize)
                .map_err(|e| CudaError::driver(e.to_string()))?;

            device.track_allocation(size);

            Ok(Self {
                device,
                size,
                allocation,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            device.track_allocation(size);

            Ok(Self {
                device,
                size,
                _ptr: 0xDEADBEEF, // Mock pointer
            })
        }
    }

    /// Get size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get raw cudarc slice (when cuda feature enabled).
    #[cfg(feature = "cuda")]
    pub fn as_slice(&self) -> &cudarc::driver::CudaSlice<u8> {
        &self.allocation
    }

    /// Get mutable cudarc slice.
    #[cfg(feature = "cuda")]
    pub fn as_slice_mut(&mut self) -> &mut cudarc::driver::CudaSlice<u8> {
        &mut self.allocation
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        self.device.track_deallocation(self.size);
    }
}

impl std::fmt::Debug for DeviceMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemory")
            .field("device", &self.device.index())
            .field("size", &self.size)
            .finish()
    }
}

// =============================================================================
// Host Memory (CPU Pageable)
// =============================================================================

/// CPU pageable memory.
///
/// Standard CPU memory allocation. Use PinnedMemory for faster
/// GPU transfers.
pub struct HostMemory {
    /// Allocated data.
    data: Vec<u8>,
    /// Size in bytes.
    size: u64,
}

impl HostMemory {
    /// Allocate host memory.
    pub fn allocate(size: u64) -> Result<Self> {
        let data = vec![0u8; size as usize];
        Ok(Self { data, size })
    }

    /// Create from existing data.
    pub fn from_vec(data: Vec<u8>) -> Self {
        let size = data.len() as u64;
        Self { data, size }
    }

    /// Get size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get data slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable data slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get raw pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// Convert to Vec, consuming the memory.
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }
}

impl std::fmt::Debug for HostMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostMemory")
            .field("size", &self.size)
            .finish()
    }
}

// =============================================================================
// Pinned Memory (CPU Pinned for DMA)
// =============================================================================

/// CPU pinned (page-locked) memory.
///
/// Pinned memory enables faster GPU transfers via DMA without
/// going through a staging buffer.
pub struct PinnedMemory {
    /// Size in bytes.
    size: u64,
    /// Mock data (when cuda feature disabled).
    #[cfg(not(feature = "cuda"))]
    data: Vec<u8>,
    /// cudarc pinned allocation.
    #[cfg(feature = "cuda")]
    allocation: cudarc::driver::CudaHostAlloc<u8>,
}

impl PinnedMemory {
    /// Allocate pinned memory.
    #[cfg(feature = "cuda")]
    pub fn allocate(device: &CudaDevice, size: u64) -> Result<Self> {
        let allocation = device
            .raw_device()
            .alloc_host::<u8>(size as usize)
            .map_err(|e| CudaError::driver(e.to_string()))?;

        Ok(Self { size, allocation })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn allocate(_device: &CudaDevice, size: u64) -> Result<Self> {
        let data = vec![0u8; size as usize];
        Ok(Self { size, data })
    }

    /// Get size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get data slice.
    #[cfg(not(feature = "cuda"))]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    #[cfg(feature = "cuda")]
    pub fn as_slice(&self) -> &[u8] {
        &self.allocation
    }

    /// Get mutable data slice.
    #[cfg(not(feature = "cuda"))]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    #[cfg(feature = "cuda")]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.allocation
    }
}

impl std::fmt::Debug for PinnedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedMemory")
            .field("size", &self.size)
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_memory() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let mem = DeviceMemory::allocate(device.clone(), 1024).unwrap();

        assert_eq!(mem.size(), 1024);
        assert_eq!(device.allocated_memory(), 1024);

        drop(mem);
        assert_eq!(device.allocated_memory(), 0);
    }

    #[test]
    fn test_host_memory() {
        let mem = HostMemory::allocate(1024).unwrap();
        assert_eq!(mem.size(), 1024);
        assert_eq!(mem.as_slice().len(), 1024);
    }

    #[test]
    fn test_host_memory_from_vec() {
        let data = vec![1u8, 2, 3, 4, 5];
        let mem = HostMemory::from_vec(data);
        assert_eq!(mem.size(), 5);
        assert_eq!(mem.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_pinned_memory() {
        let device = CudaDevice::new(0).unwrap();
        let mem = PinnedMemory::allocate(&device, 1024).unwrap();
        assert_eq!(mem.size(), 1024);
    }
}
