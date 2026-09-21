//! # Aether Bridge
//!
//! Connects Aether's state-aware planning (aether-core) with
//! CUDA execution (aether-cuda) and compute kernels (aether-kernels).
//!
//! This is the central integration point where:
//! - Decision engine outputs become physical transfers
//! - State manager locations map to GPU/CPU buffers
//! - Execution plans drive actual computation
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    AETHER BRIDGE                        │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                         │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
//! │  │   State     │    │  Transfer   │    │   Kernel    │ │
//! │  │  Executor   │◄──►│  Scheduler  │◄──►│  Executor   │ │
//! │  └─────────────┘    └─────────────┘    └─────────────┘ │
//! │         │                  │                  │        │
//! │         ▼                  ▼                  ▼        │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
//! │  │ aether-core │    │ aether-cuda │    │aether-kernel│ │
//! │  │ StateManager│    │TransferEng. │    │   Registry  │ │
//! │  └─────────────┘    └─────────────┘    └─────────────┘ │
//! │                                                         │
//! └─────────────────────────────────────────────────────────┘
//! ```

pub mod state_executor;
pub mod transfer_scheduler;
pub mod kernel_executor;
pub mod unified_runtime;
pub mod metrics;

pub use state_executor::StateExecutor;
pub use transfer_scheduler::{TransferScheduler, TransferTask, TransferPriority};
pub use kernel_executor::KernelExecutor;
pub use unified_runtime::{UnifiedRuntime, RuntimeConfig};
pub use metrics::{RuntimeMetrics, MetricsCollector};
