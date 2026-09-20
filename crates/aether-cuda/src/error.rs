//! Error types for CUDA operations.

use thiserror::Error;

/// Result type for CUDA operations.
pub type Result<T> = std::result::Result<T, CudaError>;

/// Errors that can occur in CUDA operations.
#[derive(Error, Debug)]
pub enum CudaError {
    /// CUDA driver error.
    #[error("CUDA driver error: {0}")]
    DriverError(String),

    /// Out of GPU memory.
    #[error("Out of GPU memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: u64, available: u64 },

    /// Invalid device ID.
    #[error("Invalid device ID: {0}")]
    InvalidDevice(i32),

    /// Device not initialized.
    #[error("CUDA device not initialized")]
    DeviceNotInitialized,

    /// Stream error.
    #[error("CUDA stream error: {0}")]
    StreamError(String),

    /// Transfer error.
    #[error("Memory transfer failed: {0}")]
    TransferError(String),

    /// Invalid buffer.
    #[error("Invalid buffer: {0}")]
    InvalidBuffer(String),

    /// Synchronization error.
    #[error("Synchronization error: {0}")]
    SyncError(String),

    /// Pool allocation error.
    #[error("Memory pool error: {0}")]
    PoolError(String),

    /// Feature not available.
    #[error("Feature not available: {0}")]
    NotAvailable(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl CudaError {
    /// Create an out of memory error.
    pub fn out_of_memory(requested: u64, available: u64) -> Self {
        Self::OutOfMemory {
            requested,
            available,
        }
    }

    /// Create a driver error.
    pub fn driver(msg: impl Into<String>) -> Self {
        Self::DriverError(msg.into())
    }

    /// Create a transfer error.
    pub fn transfer(msg: impl Into<String>) -> Self {
        Self::TransferError(msg.into())
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for CudaError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        Self::DriverError(err.to_string())
    }
}
