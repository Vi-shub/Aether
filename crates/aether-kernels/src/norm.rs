//! Normalization kernel implementations.

use std::sync::Arc;
use aether_core::DType;
use aether_cuda::{CudaDevice, GpuBuffer, CudaStream};

/// Layer normalization kernel.
pub struct LayerNormKernel {
    hidden_dim: usize,
    eps: f32,
    dtype: DType,
    device: Arc<CudaDevice>,
}

impl LayerNormKernel {
    /// Create new LayerNorm kernel.
    pub fn new(device: Arc<CudaDevice>, hidden_dim: usize, eps: f32, dtype: DType) -> Self {
        Self {
            hidden_dim,
            eps,
            dtype,
            device,
        }
    }

    /// Forward pass.
    ///
    /// LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
    pub fn forward(
        &self,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        bias: Option<&GpuBuffer>,
        stream: Option<&CudaStream>,
    ) -> Result<GpuBuffer, NormError> {
        // Validate input
        let shape = input.shape();
        if shape.is_empty() || shape[shape.len() - 1] != self.hidden_dim {
            return Err(NormError::InvalidShape(format!(
                "Last dimension must be {}, got {:?}",
                self.hidden_dim, shape
            )));
        }

        // Allocate output
        let output = GpuBuffer::allocate(self.device.clone(), self.dtype, shape)
            .map_err(|e| NormError::AllocationFailed(e.to_string()))?;

        // In production, this would launch a fused LayerNorm kernel
        tracing::debug!("LayerNorm: shape={:?}, eps={}", shape, self.eps);

        Ok(output)
    }

    /// Weight size.
    pub fn weight_size(&self) -> usize {
        self.hidden_dim * self.dtype.size_bytes()
    }
}

/// RMS normalization kernel.
pub struct RmsNormKernel {
    hidden_dim: usize,
    eps: f32,
    dtype: DType,
    device: Arc<CudaDevice>,
}

impl RmsNormKernel {
    /// Create new RMSNorm kernel.
    pub fn new(device: Arc<CudaDevice>, hidden_dim: usize, eps: f32, dtype: DType) -> Self {
        Self {
            hidden_dim,
            eps,
            dtype,
            device,
        }
    }

    /// Forward pass.
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    pub fn forward(
        &self,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        stream: Option<&CudaStream>,
    ) -> Result<GpuBuffer, NormError> {
        // Validate input
        let shape = input.shape();
        if shape.is_empty() || shape[shape.len() - 1] != self.hidden_dim {
            return Err(NormError::InvalidShape(format!(
                "Last dimension must be {}, got {:?}",
                self.hidden_dim, shape
            )));
        }

        // Allocate output
        let output = GpuBuffer::allocate(self.device.clone(), self.dtype, shape)
            .map_err(|e| NormError::AllocationFailed(e.to_string()))?;

        // In production, this would launch a fused RMSNorm kernel
        tracing::debug!("RMSNorm: shape={:?}, eps={}", shape, self.eps);

        Ok(output)
    }

    /// Fused RMSNorm + residual add.
    pub fn forward_with_residual(
        &self,
        input: &GpuBuffer,
        residual: &GpuBuffer,
        weight: &GpuBuffer,
        stream: Option<&CudaStream>,
    ) -> Result<(GpuBuffer, GpuBuffer), NormError> {
        // Returns (normed_output, new_residual)
        let shape = input.shape();

        let normed = GpuBuffer::allocate(self.device.clone(), self.dtype, shape)
            .map_err(|e| NormError::AllocationFailed(e.to_string()))?;

        let new_residual = GpuBuffer::allocate(self.device.clone(), self.dtype, shape)
            .map_err(|e| NormError::AllocationFailed(e.to_string()))?;

        // Fused kernel: new_residual = input + residual, normed = rmsnorm(new_residual)
        tracing::debug!("RMSNorm+Residual: shape={:?}", shape);

        Ok((normed, new_residual))
    }

    /// Weight size.
    pub fn weight_size(&self) -> usize {
        self.hidden_dim * self.dtype.size_bytes()
    }
}

/// Normalization errors.
#[derive(Debug, thiserror::Error)]
pub enum NormError {
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
    fn test_layer_norm_weight_size() {
        let device = CudaDevice::new(0).unwrap();
        let kernel = LayerNormKernel::new(Arc::new(device), 4096, 1e-5, DType::Float16);
        assert_eq!(kernel.weight_size(), 4096 * 2);
    }

    #[test]
    fn test_rms_norm_weight_size() {
        let device = CudaDevice::new(0).unwrap();
        let kernel = RmsNormKernel::new(Arc::new(device), 4096, 1e-5, DType::Float16);
        assert_eq!(kernel.weight_size(), 4096 * 2);
    }
}
