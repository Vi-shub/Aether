//! MLP (Feed-Forward Network) kernel implementations.

use std::sync::Arc;
use aether_core::DType;
use aether_cuda::{CudaDevice, GpuBuffer, CudaStream};

/// MLP configuration.
#[derive(Debug, Clone)]
pub struct MlpConfig {
    /// Hidden dimension (model dimension).
    pub hidden_dim: usize,
    /// Intermediate dimension (FFN dimension).
    pub intermediate_dim: usize,
    /// Data type.
    pub dtype: DType,
    /// Activation function.
    pub activation: Activation,
    /// Whether to use gated activation (e.g., SwiGLU).
    pub gated: bool,
}

/// Activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Gaussian Error Linear Unit.
    Gelu,
    /// Rectified Linear Unit.
    Relu,
    /// Swish/SiLU.
    Silu,
    /// Squared ReLU.
    ReluSquared,
}

impl MlpConfig {
    /// Create standard MLP config (GELU).
    pub fn standard(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            hidden_dim,
            intermediate_dim,
            dtype: DType::Float16,
            activation: Activation::Gelu,
            gated: false,
        }
    }

    /// Create LLaMA-style MLP (SwiGLU).
    pub fn swiglu(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            hidden_dim,
            intermediate_dim,
            dtype: DType::Float16,
            activation: Activation::Silu,
            gated: true,
        }
    }

    /// Get weight size for up projection.
    pub fn up_weight_size(&self) -> usize {
        let num_projections = if self.gated { 2 } else { 1 };
        self.hidden_dim * self.intermediate_dim * num_projections * self.dtype.size_bytes()
    }

    /// Get weight size for down projection.
    pub fn down_weight_size(&self) -> usize {
        self.intermediate_dim * self.hidden_dim * self.dtype.size_bytes()
    }

    /// Total weight size.
    pub fn total_weight_size(&self) -> usize {
        self.up_weight_size() + self.down_weight_size()
    }

    /// Activation memory for given batch and sequence.
    pub fn activation_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        batch_size * seq_len * self.intermediate_dim * self.dtype.size_bytes()
    }
}

/// MLP kernel.
pub struct MlpKernel {
    config: MlpConfig,
    device: Arc<CudaDevice>,
}

impl MlpKernel {
    /// Create new MLP kernel.
    pub fn new(device: Arc<CudaDevice>, config: MlpConfig) -> Self {
        Self { config, device }
    }

    /// Get configuration.
    pub fn config(&self) -> &MlpConfig {
        &self.config
    }

    /// Forward pass.
    ///
    /// For standard MLP: output = down(activation(up(x)))
    /// For gated MLP: output = down(activation(gate(x)) * up(x))
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, seq_len, hidden_dim]
    /// * `up_weight` - Up projection weight [hidden_dim, intermediate_dim]
    /// * `down_weight` - Down projection weight [intermediate_dim, hidden_dim]
    /// * `gate_weight` - Gate weight for gated MLP (optional)
    /// * `stream` - CUDA stream
    pub fn forward(
        &self,
        input: &GpuBuffer,
        up_weight: &GpuBuffer,
        down_weight: &GpuBuffer,
        gate_weight: Option<&GpuBuffer>,
        stream: Option<&CudaStream>,
    ) -> Result<GpuBuffer, MlpError> {
        // Validate inputs
        if input.shape().len() != 3 {
            return Err(MlpError::InvalidShape(
                "Expected 3D input [batch, seq, hidden]".to_string()
            ));
        }

        if self.config.gated && gate_weight.is_none() {
            return Err(MlpError::InvalidShape(
                "Gated MLP requires gate weight".to_string()
            ));
        }

        // Allocate output
        let output = GpuBuffer::allocate(
            self.device.clone(),
            self.config.dtype,
            input.shape(),
        ).map_err(|e| MlpError::AllocationFailed(e.to_string()))?;

        // In production, this would:
        // 1. GEMM: input @ up_weight -> intermediate
        // 2. Apply activation
        // 3. For gated: input @ gate_weight -> gate, then intermediate *= gate
        // 4. GEMM: intermediate @ down_weight -> output

        tracing::debug!(
            "MLP forward: batch={}, seq={}, hidden={}, intermediate={}",
            input.shape()[0],
            input.shape()[1],
            self.config.hidden_dim,
            self.config.intermediate_dim
        );

        Ok(output)
    }

    /// Estimate FLOPs.
    pub fn estimate_flops(&self, batch_size: usize, seq_len: usize) -> u64 {
        let tokens = batch_size as u64 * seq_len as u64;
        let hidden = self.config.hidden_dim as u64;
        let intermediate = self.config.intermediate_dim as u64;

        // Up projection: tokens * hidden * intermediate * 2
        // Down projection: tokens * intermediate * hidden * 2
        // Gated: double the up projection
        let multiplier = if self.config.gated { 3 } else { 2 };
        tokens * hidden * intermediate * 2 * multiplier
    }
}

/// MLP errors.
#[derive(Debug, thiserror::Error)]
pub enum MlpError {
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Kernel execution failed: {0}")]
    ExecutionFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_mlp_config() {
        let config = MlpConfig::standard(4096, 11008);
        assert!(!config.gated);
        assert_eq!(config.activation, Activation::Gelu);
    }

    #[test]
    fn test_swiglu_config() {
        let config = MlpConfig::swiglu(4096, 11008);
        assert!(config.gated);
        assert_eq!(config.activation, Activation::Silu);
    }

    #[test]
    fn test_weight_sizes() {
        let config = MlpConfig::swiglu(4096, 11008);
        // Gate + up: 4096 * 11008 * 2 * 2 bytes
        assert_eq!(config.up_weight_size(), 4096 * 11008 * 2 * 2);
        // Down: 11008 * 4096 * 2 bytes
        assert_eq!(config.down_weight_size(), 11008 * 4096 * 2);
    }
}
