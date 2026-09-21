//! Kernel registry for managing compute kernels.

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use aether_cuda::CudaDevice;

use crate::{
    AttentionKernel, AttentionConfig,
    MlpKernel, MlpConfig,
    MoeKernel, MoeConfig,
    LayerNormKernel, RmsNormKernel,
    SamplingKernel, SamplingConfig,
};
use aether_core::DType;

/// Kernel identifier.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum KernelId {
    Attention(usize),  // layer_id
    Mlp(usize),        // layer_id
    Moe(usize),        // layer_id
    LayerNorm(usize),  // layer_id
    RmsNorm(usize),    // layer_id
    Sampling,
}

/// Registered kernel types.
pub enum RegisteredKernel {
    Attention(AttentionKernel),
    Mlp(MlpKernel),
    Moe(MoeKernel),
    LayerNorm(LayerNormKernel),
    RmsNorm(RmsNormKernel),
    Sampling(SamplingKernel),
}

/// Kernel registry for managing all compute kernels.
pub struct KernelRegistry {
    device: Arc<CudaDevice>,
    kernels: RwLock<HashMap<KernelId, RegisteredKernel>>,
}

impl KernelRegistry {
    /// Create a new kernel registry.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            kernels: RwLock::new(HashMap::new()),
        }
    }

    /// Register an attention kernel.
    pub fn register_attention(&self, layer_id: usize, config: AttentionConfig) {
        let kernel = AttentionKernel::new(self.device.clone(), config);
        self.kernels.write().insert(
            KernelId::Attention(layer_id),
            RegisteredKernel::Attention(kernel),
        );
    }

    /// Register an MLP kernel.
    pub fn register_mlp(&self, layer_id: usize, config: MlpConfig) {
        let kernel = MlpKernel::new(self.device.clone(), config);
        self.kernels.write().insert(
            KernelId::Mlp(layer_id),
            RegisteredKernel::Mlp(kernel),
        );
    }

    /// Register a MoE kernel.
    pub fn register_moe(&self, layer_id: usize, config: MoeConfig) {
        let kernel = MoeKernel::new(self.device.clone(), config);
        self.kernels.write().insert(
            KernelId::Moe(layer_id),
            RegisteredKernel::Moe(kernel),
        );
    }

    /// Register a LayerNorm kernel.
    pub fn register_layer_norm(&self, layer_id: usize, hidden_dim: usize, eps: f32, dtype: DType) {
        let kernel = LayerNormKernel::new(self.device.clone(), hidden_dim, eps, dtype);
        self.kernels.write().insert(
            KernelId::LayerNorm(layer_id),
            RegisteredKernel::LayerNorm(kernel),
        );
    }

    /// Register an RMSNorm kernel.
    pub fn register_rms_norm(&self, layer_id: usize, hidden_dim: usize, eps: f32, dtype: DType) {
        let kernel = RmsNormKernel::new(self.device.clone(), hidden_dim, eps, dtype);
        self.kernels.write().insert(
            KernelId::RmsNorm(layer_id),
            RegisteredKernel::RmsNorm(kernel),
        );
    }

    /// Register a sampling kernel.
    pub fn register_sampling(&self, vocab_size: usize, config: SamplingConfig, dtype: DType) {
        let kernel = SamplingKernel::new(self.device.clone(), vocab_size, config, dtype);
        self.kernels.write().insert(
            KernelId::Sampling,
            RegisteredKernel::Sampling(kernel),
        );
    }

    /// Get an attention kernel.
    pub fn get_attention(&self, layer_id: usize) -> Option<AttentionKernelGuard> {
        let guard = self.kernels.read();
        if matches!(
            guard.get(&KernelId::Attention(layer_id)),
            Some(RegisteredKernel::Attention(_))
        ) {
            Some(AttentionKernelGuard {
                registry: self,
                layer_id,
            })
        } else {
            None
        }
    }

    /// Check if a kernel is registered.
    pub fn has_kernel(&self, id: &KernelId) -> bool {
        self.kernels.read().contains_key(id)
    }

    /// Remove a kernel.
    pub fn remove(&self, id: &KernelId) -> bool {
        self.kernels.write().remove(id).is_some()
    }

    /// Get number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.read().len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.read().is_empty()
    }

    /// Register all kernels for a transformer model.
    pub fn register_transformer(
        &self,
        num_layers: usize,
        attention_config: AttentionConfig,
        mlp_config: Option<MlpConfig>,
        moe_config: Option<MoeConfig>,
        norm_hidden_dim: usize,
        norm_eps: f32,
        dtype: DType,
    ) {
        for layer_id in 0..num_layers {
            // Attention
            self.register_attention(layer_id, attention_config.clone());

            // FFN: either MLP or MoE
            if let Some(ref config) = moe_config {
                self.register_moe(layer_id, config.clone());
            } else if let Some(ref config) = mlp_config {
                self.register_mlp(layer_id, config.clone());
            }

            // Norms (2 per layer typically)
            self.register_rms_norm(layer_id * 2, norm_hidden_dim, norm_eps, dtype);
            self.register_rms_norm(layer_id * 2 + 1, norm_hidden_dim, norm_eps, dtype);
        }
    }
}

/// Guard for accessing attention kernel.
pub struct AttentionKernelGuard<'a> {
    registry: &'a KernelRegistry,
    layer_id: usize,
}

impl<'a> AttentionKernelGuard<'a> {
    /// Execute a function with the attention kernel.
    pub fn with<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&AttentionKernel) -> R,
    {
        let guard = self.registry.kernels.read();
        if let Some(RegisteredKernel::Attention(kernel)) =
            guard.get(&KernelId::Attention(self.layer_id))
        {
            Some(f(kernel))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let device = CudaDevice::new(0).unwrap();
        let registry = KernelRegistry::new(Arc::new(device));
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_attention() {
        let device = CudaDevice::new(0).unwrap();
        let registry = KernelRegistry::new(Arc::new(device));

        let config = AttentionConfig::mha(32, 128, 4096);
        registry.register_attention(0, config);

        assert!(registry.has_kernel(&KernelId::Attention(0)));
        assert!(!registry.has_kernel(&KernelId::Attention(1)));
    }
}
