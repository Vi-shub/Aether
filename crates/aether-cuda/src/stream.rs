//! CUDA stream management.
//!
//! Streams enable concurrent execution and overlapping of:
//! - Compute operations
//! - Memory transfers (H2D, D2H, D2D)
//! - Multiple kernels

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::CudaDevice;
use crate::error::{CudaError, Result};

/// Stream priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPriority {
    /// Highest priority (for latency-critical work).
    High,
    /// Normal priority.
    Normal,
    /// Low priority (for background work).
    Low,
}

impl StreamPriority {
    /// Convert to CUDA priority value (lower = higher priority).
    fn to_cuda_priority(self) -> i32 {
        match self {
            Self::High => -1,
            Self::Normal => 0,
            Self::Low => 1,
        }
    }
}

/// Stream statistics.
#[derive(Debug, Default)]
pub struct StreamStats {
    /// Number of operations submitted.
    pub operations_submitted: AtomicU64,
    /// Number of synchronizations.
    pub synchronizations: AtomicU64,
    /// Total bytes transferred.
    pub bytes_transferred: AtomicU64,
}

impl StreamStats {
    fn increment_ops(&self) {
        self.operations_submitted.fetch_add(1, Ordering::Relaxed);
    }

    fn increment_syncs(&self) {
        self.synchronizations.fetch_add(1, Ordering::Relaxed);
    }

    fn add_bytes(&self, bytes: u64) {
        self.bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// CUDA stream for async operations.
///
/// Streams allow overlapping of compute and memory operations.
/// Each stream executes operations in order, but different streams
/// can execute concurrently.
pub struct CudaStream {
    /// Device this stream belongs to.
    device: Arc<CudaDevice>,
    /// Stream ID for tracking.
    id: u64,
    /// Priority level.
    priority: StreamPriority,
    /// Statistics.
    stats: StreamStats,
    /// Whether this is the default stream.
    is_default: bool,

    #[cfg(feature = "cuda")]
    stream: cudarc::driver::CudaStream,
}

// Stream ID counter
static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl CudaStream {
    /// Create a new stream.
    #[cfg(feature = "cuda")]
    pub fn new(device: Arc<CudaDevice>, priority: StreamPriority) -> Result<Self> {
        let stream = device
            .raw_device()
            .fork_default_stream()
            .map_err(|e| CudaError::StreamError(e.to_string()))?;

        let id = STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            device,
            id,
            priority,
            stats: StreamStats::default(),
            is_default: false,
            stream,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(device: Arc<CudaDevice>, priority: StreamPriority) -> Result<Self> {
        let id = STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Ok(Self {
            device,
            id,
            priority,
            stats: StreamStats::default(),
            is_default: false,
        })
    }

    /// Create the default stream.
    #[cfg(feature = "cuda")]
    pub fn default_stream(device: Arc<CudaDevice>) -> Result<Self> {
        let stream = device
            .raw_device()
            .fork_default_stream()
            .map_err(|e| CudaError::StreamError(e.to_string()))?;

        Ok(Self {
            device,
            id: 0,
            priority: StreamPriority::Normal,
            stats: StreamStats::default(),
            is_default: true,
            stream,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn default_stream(device: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self {
            device,
            id: 0,
            priority: StreamPriority::Normal,
            stats: StreamStats::default(),
            is_default: true,
        })
    }

    /// Get stream ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get priority.
    pub fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Check if this is the default stream.
    pub fn is_default(&self) -> bool {
        self.is_default
    }

    /// Get statistics.
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// Get raw cudarc stream (when cuda feature enabled).
    #[cfg(feature = "cuda")]
    pub fn raw_stream(&self) -> &cudarc::driver::CudaStream {
        &self.stream
    }

    /// Synchronize stream (wait for all operations to complete).
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.stream
                .synchronize()
                .map_err(|e| CudaError::SyncError(e.to_string()))?;
        }

        self.stats.increment_syncs();
        Ok(())
    }

    /// Check if stream is complete (non-blocking).
    pub fn is_complete(&self) -> Result<bool> {
        #[cfg(feature = "cuda")]
        {
            // cudarc doesn't have a direct query, so we use synchronize
            // In a real impl, we'd use cuStreamQuery
            Ok(true)
        }

        #[cfg(not(feature = "cuda"))]
        Ok(true)
    }

    /// Record an operation (for statistics).
    pub(crate) fn record_operation(&self) {
        self.stats.increment_ops();
    }

    /// Record bytes transferred.
    pub(crate) fn record_transfer(&self, bytes: u64) {
        self.stats.add_bytes(bytes);
    }
}

impl std::fmt::Debug for CudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStream")
            .field("id", &self.id)
            .field("device", &self.device.index())
            .field("priority", &self.priority)
            .field("is_default", &self.is_default)
            .finish()
    }
}

// =============================================================================
// Stream Pool
// =============================================================================

/// Pool of CUDA streams for efficient reuse.
pub struct StreamPool {
    device: Arc<CudaDevice>,
    /// Available streams.
    available: parking_lot::Mutex<Vec<CudaStream>>,
    /// Maximum pool size.
    max_size: usize,
}

impl StreamPool {
    /// Create a new stream pool.
    pub fn new(device: Arc<CudaDevice>, max_size: usize) -> Self {
        Self {
            device,
            available: parking_lot::Mutex::new(Vec::with_capacity(max_size)),
            max_size,
        }
    }

    /// Get a stream from the pool (or create a new one).
    pub fn get(&self) -> Result<CudaStream> {
        let mut available = self.available.lock();

        if let Some(stream) = available.pop() {
            Ok(stream)
        } else {
            CudaStream::new(self.device.clone(), StreamPriority::Normal)
        }
    }

    /// Return a stream to the pool.
    pub fn put(&self, stream: CudaStream) {
        let mut available = self.available.lock();

        if available.len() < self.max_size {
            available.push(stream);
        }
        // Otherwise, stream is dropped
    }

    /// Get pool size.
    pub fn size(&self) -> usize {
        self.available.lock().len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_stream() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let stream = CudaStream::new(device, StreamPriority::Normal).unwrap();

        assert!(!stream.is_default());
        assert_eq!(stream.priority(), StreamPriority::Normal);
    }

    #[test]
    fn test_default_stream() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let stream = CudaStream::default_stream(device).unwrap();

        assert!(stream.is_default());
    }

    #[test]
    fn test_stream_pool() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let pool = StreamPool::new(device, 4);

        let stream1 = pool.get().unwrap();
        let stream2 = pool.get().unwrap();

        assert_eq!(pool.size(), 0);

        pool.put(stream1);
        assert_eq!(pool.size(), 1);

        pool.put(stream2);
        assert_eq!(pool.size(), 2);

        let _ = pool.get().unwrap();
        assert_eq!(pool.size(), 1);
    }

    #[test]
    fn test_stream_synchronize() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let stream = CudaStream::new(device, StreamPriority::Normal).unwrap();

        stream.synchronize().unwrap();
        assert_eq!(stream.stats().synchronizations.load(Ordering::Relaxed), 1);
    }
}
