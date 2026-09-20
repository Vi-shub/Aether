//! # Aether CUDA
//!
//! CUDA memory operations and async transfer engine for Aether runtime.
//!
//! This crate provides:
//! - GPU memory allocation and management
//! - Async memory transfers between memory tiers
//! - CUDA stream management for overlapping compute/transfer
//! - Memory pools for efficient allocation
//!
//! ## Features
//!
//! - `cuda`: Enable actual CUDA operations (requires CUDA toolkit)
//! - `mock`: Use mock implementations for testing without GPU
//!
//! ## Example
//!
//! ```rust,ignore
//! use aether_cuda::{CudaDevice, MemoryPool, TransferEngine};
//!
//! // Initialize device
//! let device = CudaDevice::new(0)?;
//!
//! // Create memory pool
//! let pool = MemoryPool::new(&device, 8 * 1024 * 1024 * 1024)?; // 8GB
//!
//! // Allocate tensor
//! let buffer = pool.allocate(1024 * 1024)?; // 1MB
//!
//! // Create transfer engine
//! let engine = TransferEngine::new(&device)?;
//!
//! // Async transfer
//! engine.copy_host_to_device_async(&host_data, &buffer)?;
//! engine.synchronize()?;
//! ```

pub mod device;
pub mod memory;
pub mod transfer;
pub mod stream;
pub mod buffer;
pub mod pool;
pub mod error;

#[cfg(feature = "mock")]
pub mod mock;

pub use device::CudaDevice;
pub use memory::{DeviceMemory, HostMemory, PinnedMemory};
pub use transfer::TransferEngine;
pub use stream::CudaStream;
pub use buffer::{GpuBuffer, CpuBuffer};
pub use pool::MemoryPool;
pub use error::{CudaError, Result};

/// Re-export core types
pub use aether_core::{Location, StateId};
