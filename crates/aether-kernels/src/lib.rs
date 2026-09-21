//! # Aether Kernels
//!
//! Compute kernels for Aether runtime including:
//! - Attention (FlashAttention-compatible)
//! - MLP (feed-forward networks)
//! - MoE (mixture of experts)
//! - LayerNorm, RMSNorm
//! - Sampling
//!
//! These kernels integrate with Aether's state-aware execution model,
//! enabling overlapped compute and memory operations.

pub mod attention;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod sampling;
pub mod kernel_registry;

pub use attention::{AttentionKernel, AttentionConfig, AttentionOutput};
pub use mlp::{MlpKernel, MlpConfig};
pub use moe::{MoeKernel, MoeConfig, ExpertSelection};
pub use norm::{LayerNormKernel, RmsNormKernel};
pub use sampling::{SamplingKernel, SamplingConfig};
pub use kernel_registry::KernelRegistry;
