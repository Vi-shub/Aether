//! Async transfer engine.
//!
//! Handles memory transfers between:
//! - Host (CPU) <-> Device (GPU)
//! - Device <-> Device (multi-GPU)
//! - With support for overlapping with compute.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::VecDeque;
use parking_lot::Mutex;

use aether_core::Location;

use crate::device::CudaDevice;
use crate::stream::{CudaStream, StreamPriority, StreamPool};
use crate::memory::{DeviceMemory, HostMemory, PinnedMemory};
use crate::buffer::{GpuBuffer, CpuBuffer};
use crate::error::{CudaError, Result};

// =============================================================================
// Transfer Direction
// =============================================================================

/// Direction of memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to Device (CPU -> GPU).
    HostToDevice,
    /// Device to Host (GPU -> CPU).
    DeviceToHost,
    /// Device to Device (GPU -> GPU, same or different).
    DeviceToDevice,
}

impl TransferDirection {
    /// Convert from Location pair.
    pub fn from_locations(src: Location, dst: Location) -> Option<Self> {
        match (src, dst) {
            (Location::Dram, Location::Hbm) => Some(Self::HostToDevice),
            (Location::Hbm, Location::Dram) => Some(Self::DeviceToHost),
            (Location::Hbm, Location::Hbm) => Some(Self::DeviceToDevice),
            _ => None,
        }
    }
}

// =============================================================================
// Transfer Request
// =============================================================================

/// A pending transfer request.
#[derive(Debug)]
pub struct TransferRequest {
    /// Unique ID for tracking.
    pub id: u64,
    /// Direction.
    pub direction: TransferDirection,
    /// Size in bytes.
    pub size: u64,
    /// Associated state ID (if any).
    pub state_id: Option<aether_core::StateId>,
    /// Stream this transfer is on.
    pub stream_id: u64,
    /// Whether transfer is complete.
    pub complete: bool,
}

/// Handle to a pending async transfer.
pub struct TransferHandle {
    request_id: u64,
    stream: Arc<CudaStream>,
}

impl TransferHandle {
    /// Wait for transfer to complete.
    pub fn wait(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Check if transfer is complete (non-blocking).
    pub fn is_complete(&self) -> Result<bool> {
        self.stream.is_complete()
    }

    /// Get request ID.
    pub fn id(&self) -> u64 {
        self.request_id
    }
}

// =============================================================================
// Transfer Statistics
// =============================================================================

/// Transfer engine statistics.
#[derive(Debug, Default)]
pub struct TransferStats {
    /// Total host-to-device transfers.
    pub h2d_transfers: AtomicU64,
    /// Total device-to-host transfers.
    pub d2h_transfers: AtomicU64,
    /// Total device-to-device transfers.
    pub d2d_transfers: AtomicU64,
    /// Total bytes transferred H2D.
    pub h2d_bytes: AtomicU64,
    /// Total bytes transferred D2H.
    pub d2h_bytes: AtomicU64,
    /// Total bytes transferred D2D.
    pub d2d_bytes: AtomicU64,
}

impl TransferStats {
    fn record(&self, direction: TransferDirection, bytes: u64) {
        match direction {
            TransferDirection::HostToDevice => {
                self.h2d_transfers.fetch_add(1, Ordering::Relaxed);
                self.h2d_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
            TransferDirection::DeviceToHost => {
                self.d2h_transfers.fetch_add(1, Ordering::Relaxed);
                self.d2h_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
            TransferDirection::DeviceToDevice => {
                self.d2d_transfers.fetch_add(1, Ordering::Relaxed);
                self.d2d_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
        }
    }

    /// Total transfers.
    pub fn total_transfers(&self) -> u64 {
        self.h2d_transfers.load(Ordering::Relaxed)
            + self.d2h_transfers.load(Ordering::Relaxed)
            + self.d2d_transfers.load(Ordering::Relaxed)
    }

    /// Total bytes transferred.
    pub fn total_bytes(&self) -> u64 {
        self.h2d_bytes.load(Ordering::Relaxed)
            + self.d2h_bytes.load(Ordering::Relaxed)
            + self.d2d_bytes.load(Ordering::Relaxed)
    }
}

// =============================================================================
// Transfer Engine
// =============================================================================

/// Async transfer engine for GPU memory operations.
///
/// The transfer engine manages:
/// - Async memory transfers via CUDA streams
/// - Stream pool for efficient reuse
/// - Transfer tracking and statistics
/// - Overlapping transfers with compute
pub struct TransferEngine {
    /// Primary device.
    device: Arc<CudaDevice>,
    /// Stream pool for transfers.
    stream_pool: StreamPool,
    /// Default transfer stream.
    default_stream: Arc<CudaStream>,
    /// Pending transfers.
    pending: Mutex<VecDeque<TransferRequest>>,
    /// Transfer ID counter.
    transfer_id: AtomicU64,
    /// Statistics.
    pub stats: TransferStats,
}

impl TransferEngine {
    /// Create a new transfer engine.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let stream_pool = StreamPool::new(device.clone(), 8);
        let default_stream = Arc::new(CudaStream::new(device.clone(), StreamPriority::High)?);

        Ok(Self {
            device,
            stream_pool,
            default_stream,
            pending: Mutex::new(VecDeque::new()),
            transfer_id: AtomicU64::new(1),
            stats: TransferStats::default(),
        })
    }

    /// Get device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get next transfer ID.
    fn next_id(&self) -> u64 {
        self.transfer_id.fetch_add(1, Ordering::Relaxed)
    }

    // =========================================================================
    // Host <-> Device Transfers
    // =========================================================================

    /// Copy from host to device (synchronous).
    pub fn copy_h2d(&self, src: &[u8], dst: &mut DeviceMemory) -> Result<()> {
        if src.len() as u64 > dst.size() {
            return Err(CudaError::transfer("Source larger than destination"));
        }

        #[cfg(feature = "cuda")]
        {
            self.device
                .raw_device()
                .htod_copy_into(src, dst.as_slice_mut())
                .map_err(|e| CudaError::transfer(e.to_string()))?;
        }

        self.stats.record(TransferDirection::HostToDevice, src.len() as u64);
        Ok(())
    }

    /// Copy from device to host (synchronous).
    pub fn copy_d2h(&self, src: &DeviceMemory, dst: &mut [u8]) -> Result<()> {
        if dst.len() as u64 < src.size() {
            return Err(CudaError::transfer("Destination smaller than source"));
        }

        #[cfg(feature = "cuda")]
        {
            self.device
                .raw_device()
                .dtoh_sync_copy_into(src.as_slice(), dst)
                .map_err(|e| CudaError::transfer(e.to_string()))?;
        }

        self.stats.record(TransferDirection::DeviceToHost, src.size());
        Ok(())
    }

    /// Copy from host to device (async).
    pub fn copy_h2d_async(
        &self,
        src: &[u8],
        dst: &mut DeviceMemory,
        stream: Option<&CudaStream>,
    ) -> Result<TransferHandle> {
        if src.len() as u64 > dst.size() {
            return Err(CudaError::transfer("Source larger than destination"));
        }

        let stream = stream
            .map(|s| Arc::new(CudaStream::new(self.device.clone(), StreamPriority::Normal).unwrap()))
            .unwrap_or_else(|| self.default_stream.clone());

        #[cfg(feature = "cuda")]
        {
            // In cudarc, async copy would use stream.
            // For now, we do sync copy (cudarc doesn't expose async easily)
            self.device
                .raw_device()
                .htod_copy_into(src, dst.as_slice_mut())
                .map_err(|e| CudaError::transfer(e.to_string()))?;
        }

        let id = self.next_id();
        stream.record_transfer(src.len() as u64);
        self.stats.record(TransferDirection::HostToDevice, src.len() as u64);

        let request = TransferRequest {
            id,
            direction: TransferDirection::HostToDevice,
            size: src.len() as u64,
            state_id: None,
            stream_id: stream.id(),
            complete: true, // Sync for now
        };

        self.pending.lock().push_back(request);

        Ok(TransferHandle {
            request_id: id,
            stream,
        })
    }

    /// Copy from device to host (async).
    pub fn copy_d2h_async(
        &self,
        src: &DeviceMemory,
        dst: &mut [u8],
        stream: Option<&CudaStream>,
    ) -> Result<TransferHandle> {
        if dst.len() as u64 < src.size() {
            return Err(CudaError::transfer("Destination smaller than source"));
        }

        let stream = stream
            .map(|s| Arc::new(CudaStream::new(self.device.clone(), StreamPriority::Normal).unwrap()))
            .unwrap_or_else(|| self.default_stream.clone());

        #[cfg(feature = "cuda")]
        {
            self.device
                .raw_device()
                .dtoh_sync_copy_into(src.as_slice(), dst)
                .map_err(|e| CudaError::transfer(e.to_string()))?;
        }

        let id = self.next_id();
        stream.record_transfer(src.size());
        self.stats.record(TransferDirection::DeviceToHost, src.size());

        let request = TransferRequest {
            id,
            direction: TransferDirection::DeviceToHost,
            size: src.size(),
            state_id: None,
            stream_id: stream.id(),
            complete: true,
        };

        self.pending.lock().push_back(request);

        Ok(TransferHandle {
            request_id: id,
            stream,
        })
    }

    // =========================================================================
    // Buffer Transfers
    // =========================================================================

    /// Upload CPU buffer to GPU.
    pub fn upload(&self, src: &CpuBuffer, dst: &mut GpuBuffer) -> Result<()> {
        if src.size_bytes() > dst.size_bytes() {
            return Err(CudaError::transfer("Source larger than destination"));
        }

        self.copy_h2d(src.as_slice(), dst.memory_mut())
    }

    /// Download GPU buffer to CPU.
    pub fn download(&self, src: &GpuBuffer, dst: &mut CpuBuffer) -> Result<()> {
        if src.size_bytes() > dst.size_bytes() {
            return Err(CudaError::transfer("Source larger than destination"));
        }

        self.copy_d2h(src.memory(), dst.as_mut_slice())
    }

    /// Upload CPU buffer to GPU (async).
    pub fn upload_async(
        &self,
        src: &CpuBuffer,
        dst: &mut GpuBuffer,
    ) -> Result<TransferHandle> {
        if src.size_bytes() > dst.size_bytes() {
            return Err(CudaError::transfer("Source larger than destination"));
        }

        self.copy_h2d_async(src.as_slice(), dst.memory_mut(), None)
    }

    /// Download GPU buffer to CPU (async).
    pub fn download_async(
        &self,
        src: &GpuBuffer,
        dst: &mut CpuBuffer,
    ) -> Result<TransferHandle> {
        if src.size_bytes() > dst.size_bytes() {
            return Err(CudaError::transfer("Source larger than destination"));
        }

        self.copy_d2h_async(src.memory(), dst.as_mut_slice(), None)
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    /// Synchronize the default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.default_stream.synchronize()
    }

    /// Synchronize all streams.
    pub fn synchronize_all(&self) -> Result<()> {
        self.device.synchronize()
    }

    /// Get number of pending transfers.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Clear completed transfers from pending list.
    pub fn cleanup_completed(&self) {
        let mut pending = self.pending.lock();
        pending.retain(|t| !t.complete);
    }
}

impl std::fmt::Debug for TransferEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferEngine")
            .field("device", &self.device.index())
            .field("pending", &self.pending_count())
            .field("total_transfers", &self.stats.total_transfers())
            .field("total_bytes", &self.stats.total_bytes())
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
    fn test_transfer_direction() {
        assert_eq!(
            TransferDirection::from_locations(Location::Dram, Location::Hbm),
            Some(TransferDirection::HostToDevice)
        );
        assert_eq!(
            TransferDirection::from_locations(Location::Hbm, Location::Dram),
            Some(TransferDirection::DeviceToHost)
        );
        assert_eq!(
            TransferDirection::from_locations(Location::Hbm, Location::Hbm),
            Some(TransferDirection::DeviceToDevice)
        );
    }

    #[test]
    fn test_transfer_engine_creation() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let engine = TransferEngine::new(device).unwrap();

        assert_eq!(engine.pending_count(), 0);
        assert_eq!(engine.stats.total_transfers(), 0);
    }

    #[test]
    fn test_transfer_stats() {
        let stats = TransferStats::default();

        stats.record(TransferDirection::HostToDevice, 1024);
        stats.record(TransferDirection::HostToDevice, 2048);
        stats.record(TransferDirection::DeviceToHost, 512);

        assert_eq!(stats.h2d_transfers.load(Ordering::Relaxed), 2);
        assert_eq!(stats.h2d_bytes.load(Ordering::Relaxed), 3072);
        assert_eq!(stats.d2h_transfers.load(Ordering::Relaxed), 1);
        assert_eq!(stats.d2h_bytes.load(Ordering::Relaxed), 512);
        assert_eq!(stats.total_transfers(), 3);
        assert_eq!(stats.total_bytes(), 3584);
    }

    #[test]
    fn test_h2d_transfer() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let engine = TransferEngine::new(device.clone()).unwrap();

        let src_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut dst = DeviceMemory::allocate(device, 1024).unwrap();

        engine.copy_h2d(&src_data, &mut dst).unwrap();

        assert_eq!(engine.stats.h2d_transfers.load(Ordering::Relaxed), 1);
        assert_eq!(engine.stats.h2d_bytes.load(Ordering::Relaxed), 8);
    }
}
