//! CUDA device management.

use std::sync::Arc;
use parking_lot::RwLock;

use crate::error::{CudaError, Result};

/// Information about a CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index.
    pub index: i32,
    /// Device name.
    pub name: String,
    /// Total memory in bytes.
    pub total_memory: u64,
    /// Free memory in bytes.
    pub free_memory: u64,
    /// Compute capability (major, minor).
    pub compute_capability: (i32, i32),
    /// Number of SMs.
    pub sm_count: i32,
    /// Memory bus width in bits.
    pub memory_bus_width: i32,
    /// Memory clock rate in kHz.
    pub memory_clock_rate: i32,
    /// Peak memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
}

impl DeviceInfo {
    /// Check if device supports async memory operations.
    pub fn supports_async_copy(&self) -> bool {
        // Compute capability 2.0+ supports async copy
        self.compute_capability.0 >= 2
    }

    /// Check if device supports unified memory.
    pub fn supports_unified_memory(&self) -> bool {
        // Compute capability 3.0+ supports unified memory
        self.compute_capability.0 >= 3
    }

    /// Check if device supports concurrent kernels.
    pub fn supports_concurrent_kernels(&self) -> bool {
        self.compute_capability.0 >= 2
    }
}

/// CUDA device handle.
pub struct CudaDevice {
    /// Device index.
    index: i32,
    /// Device information.
    info: DeviceInfo,
    /// Current allocated memory.
    allocated_memory: Arc<RwLock<u64>>,

    /// cudarc device handle (when cuda feature enabled).
    #[cfg(feature = "cuda")]
    device: Arc<cudarc::driver::CudaDevice>,
}

impl CudaDevice {
    /// Initialize a CUDA device.
    #[cfg(feature = "cuda")]
    pub fn new(device_index: i32) -> Result<Self> {
        use cudarc::driver::CudaDevice as CudarcDevice;

        let device = CudarcDevice::new(device_index as usize)
            .map_err(|e| CudaError::driver(format!("Failed to init device {}: {}", device_index, e)))?;

        let device = Arc::new(device);

        // Get device properties
        let name = device.name()
            .map_err(|e| CudaError::driver(e.to_string()))?;

        let (free, total) = device.memory_info()
            .map_err(|e| CudaError::driver(e.to_string()))?;

        let info = DeviceInfo {
            index: device_index,
            name,
            total_memory: total as u64,
            free_memory: free as u64,
            compute_capability: (8, 0), // Default, would get from actual device
            sm_count: 108, // Default for A100
            memory_bus_width: 5120, // A100
            memory_clock_rate: 1215000, // A100
            memory_bandwidth_gbps: 2039.0, // A100
        };

        Ok(Self {
            index: device_index,
            info,
            allocated_memory: Arc::new(RwLock::new(0)),
            device,
        })
    }

    /// Initialize a mock CUDA device (for testing without GPU).
    #[cfg(not(feature = "cuda"))]
    pub fn new(device_index: i32) -> Result<Self> {
        let info = DeviceInfo {
            index: device_index,
            name: format!("Mock GPU {}", device_index),
            total_memory: 80 * 1024 * 1024 * 1024, // 80GB
            free_memory: 80 * 1024 * 1024 * 1024,
            compute_capability: (8, 0),
            sm_count: 108,
            memory_bus_width: 5120,
            memory_clock_rate: 1215000,
            memory_bandwidth_gbps: 2039.0,
        };

        Ok(Self {
            index: device_index,
            info,
            allocated_memory: Arc::new(RwLock::new(0)),
        })
    }

    /// Get device index.
    pub fn index(&self) -> i32 {
        self.index
    }

    /// Get device information.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get total memory.
    pub fn total_memory(&self) -> u64 {
        self.info.total_memory
    }

    /// Get current free memory.
    pub fn free_memory(&self) -> u64 {
        let allocated = *self.allocated_memory.read();
        self.info.total_memory.saturating_sub(allocated)
    }

    /// Get allocated memory.
    pub fn allocated_memory(&self) -> u64 {
        *self.allocated_memory.read()
    }

    /// Track memory allocation.
    pub(crate) fn track_allocation(&self, bytes: u64) {
        *self.allocated_memory.write() += bytes;
    }

    /// Track memory deallocation.
    pub(crate) fn track_deallocation(&self, bytes: u64) {
        let mut allocated = self.allocated_memory.write();
        *allocated = allocated.saturating_sub(bytes);
    }

    /// Check if allocation is possible.
    pub fn can_allocate(&self, bytes: u64) -> bool {
        bytes <= self.free_memory()
    }

    /// Get the underlying cudarc device (if available).
    #[cfg(feature = "cuda")]
    pub fn raw_device(&self) -> &Arc<cudarc::driver::CudaDevice> {
        &self.device
    }

    /// Synchronize device.
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.device.synchronize()
                .map_err(|e| CudaError::SyncError(e.to_string()))?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("index", &self.index)
            .field("name", &self.info.name)
            .field("total_memory", &self.info.total_memory)
            .field("allocated", &self.allocated_memory())
            .finish()
    }
}

/// Get number of available CUDA devices.
#[cfg(feature = "cuda")]
pub fn device_count() -> Result<i32> {
    cudarc::driver::result::device::count()
        .map(|c| c as i32)
        .map_err(|e| CudaError::driver(e.to_string()))
}

#[cfg(not(feature = "cuda"))]
pub fn device_count() -> Result<i32> {
    Ok(1) // Mock: 1 device
}

/// Get all available devices.
pub fn get_all_devices() -> Result<Vec<CudaDevice>> {
    let count = device_count()?;
    let mut devices = Vec::with_capacity(count as usize);
    
    for i in 0..count {
        devices.push(CudaDevice::new(i)?);
    }
    
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_device() {
        let device = CudaDevice::new(0).unwrap();
        assert_eq!(device.index(), 0);
        assert!(device.total_memory() > 0);
        assert!(device.can_allocate(1024));
    }

    #[test]
    fn test_memory_tracking() {
        let device = CudaDevice::new(0).unwrap();
        let initial_free = device.free_memory();

        device.track_allocation(1024);
        assert_eq!(device.allocated_memory(), 1024);
        assert_eq!(device.free_memory(), initial_free - 1024);

        device.track_deallocation(512);
        assert_eq!(device.allocated_memory(), 512);
    }
}
