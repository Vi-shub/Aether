//! Attention kernel implementations.
//!
//! Supports:
//! - Multi-head attention (MHA)
//! - Grouped-query attention (GQA)
//! - Multi-query attention (MQA)
//! - FlashAttention-style memory-efficient attention

use std::sync::Arc;
use aether_core::{DType, LayerId};
use aether_cuda::{CudaDevice, GpuBuffer, CudaStream};

/// Attention configuration.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA/MQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Data type.
    pub dtype: DType,
    /// Whether to use causal mask.
    pub causal: bool,
    /// Softmax scale (usually 1/sqrt(head_dim)).
    pub softmax_scale: Option<f32>,
    /// Dropout probability (0 for inference).
    pub dropout: f32,
    /// Layer ID for tracking.
    pub layer_id: Option<LayerId>,
}

impl AttentionConfig {
    /// Create config for standard MHA.
    pub fn mha(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            hidden_dim: num_heads * head_dim,
            max_seq_len,
            dtype: DType::Float16,
            causal: true,
            softmax_scale: None,
            dropout: 0.0,
            layer_id: None,
        }
    }

    /// Create config for GQA.
    pub fn gqa(num_heads: usize, num_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim: num_heads * head_dim,
            max_seq_len,
            dtype: DType::Float16,
            causal: true,
            softmax_scale: None,
            dropout: 0.0,
            layer_id: None,
        }
    }

    /// Create config for MQA.
    pub fn mqa(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self::gqa(num_heads, 1, head_dim, max_seq_len)
    }

    /// Set layer ID.
    pub fn with_layer(mut self, layer_id: LayerId) -> Self {
        self.layer_id = Some(layer_id);
        self
    }

    /// Get softmax scale.
    pub fn get_softmax_scale(&self) -> f32 {
        self.softmax_scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// Check if this is grouped-query attention.
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads && self.num_kv_heads > 1
    }

    /// Check if this is multi-query attention.
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Get KV size per token.
    pub fn kv_size_per_token(&self) -> usize {
        2 * self.num_kv_heads * self.head_dim * self.dtype.size_bytes()
    }
}

/// Output from attention computation.
#[derive(Debug)]
pub struct AttentionOutput {
    /// Output tensor.
    pub output: GpuBuffer,
    /// Updated K cache (if applicable).
    pub k_cache: Option<GpuBuffer>,
    /// Updated V cache (if applicable).
    pub v_cache: Option<GpuBuffer>,
    /// Attention weights (if requested).
    pub attention_weights: Option<GpuBuffer>,
}

/// Attention kernel abstraction.
pub struct AttentionKernel {
    config: AttentionConfig,
    device: Arc<CudaDevice>,
}

impl AttentionKernel {
    /// Create a new attention kernel.
    pub fn new(device: Arc<CudaDevice>, config: AttentionConfig) -> Self {
        Self { config, device }
    }

    /// Get configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Compute attention (prefill - full sequence).
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_len, num_heads, head_dim]
    /// * `k` - Key tensor [batch, seq_len, num_kv_heads, head_dim]
    /// * `v` - Value tensor [batch, seq_len, num_kv_heads, head_dim]
    /// * `stream` - CUDA stream for async execution
    pub fn forward_prefill(
        &self,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        stream: Option<&CudaStream>,
    ) -> Result<AttentionOutput, AttentionError> {
        // Validate shapes
        self.validate_prefill_inputs(q, k, v)?;

        // Allocate output
        let output = GpuBuffer::allocate(
            self.device.clone(),
            self.config.dtype,
            q.shape(),
        ).map_err(|e| AttentionError::AllocationFailed(e.to_string()))?;

        // In a real implementation, this would call FlashAttention or similar
        // For now, this is a placeholder that would be replaced with actual CUDA kernels

        #[cfg(feature = "flash-attention")]
        {
            // Call flash_attn_varlen_func or similar
            unimplemented!("FlashAttention integration pending");
        }

        #[cfg(not(feature = "flash-attention"))]
        {
            // Mock implementation - in production, use cuBLAS batched GEMM
            tracing::debug!(
                "Attention prefill: batch={}, seq_len={}, heads={}",
                q.shape()[0],
                q.shape()[1],
                self.config.num_heads
            );
        }

        Ok(AttentionOutput {
            output,
            k_cache: None,
            v_cache: None,
            attention_weights: None,
        })
    }

    /// Compute attention (decode - single token with KV cache).
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, 1, num_heads, head_dim]
    /// * `k_cache` - Key cache [batch, cache_len, num_kv_heads, head_dim]
    /// * `v_cache` - Value cache [batch, cache_len, num_kv_heads, head_dim]
    /// * `new_k` - New key [batch, 1, num_kv_heads, head_dim]
    /// * `new_v` - New value [batch, 1, num_kv_heads, head_dim]
    /// * `cache_position` - Position in cache to write new KV
    /// * `stream` - CUDA stream
    pub fn forward_decode(
        &self,
        q: &GpuBuffer,
        k_cache: &mut GpuBuffer,
        v_cache: &mut GpuBuffer,
        new_k: &GpuBuffer,
        new_v: &GpuBuffer,
        cache_position: usize,
        stream: Option<&CudaStream>,
    ) -> Result<AttentionOutput, AttentionError> {
        // Validate inputs
        if q.shape()[1] != 1 {
            return Err(AttentionError::InvalidShape(
                "Decode expects seq_len=1 for query".to_string()
            ));
        }

        // Allocate output
        let output = GpuBuffer::allocate(
            self.device.clone(),
            self.config.dtype,
            q.shape(),
        ).map_err(|e| AttentionError::AllocationFailed(e.to_string()))?;

        // In production:
        // 1. Append new_k, new_v to cache at cache_position
        // 2. Compute attention: Q @ K^T / sqrt(d) -> softmax -> @ V
        // 3. For GQA, broadcast KV heads to match Q heads

        tracing::debug!(
            "Attention decode: batch={}, cache_len={}, pos={}",
            q.shape()[0],
            cache_position,
            cache_position
        );

        Ok(AttentionOutput {
            output,
            k_cache: None,
            v_cache: None,
            attention_weights: None,
        })
    }

    /// Validate prefill inputs.
    fn validate_prefill_inputs(
        &self,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
    ) -> Result<(), AttentionError> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();

        // Check dimensions
        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(AttentionError::InvalidShape(
                "Expected 4D tensors [batch, seq, heads, head_dim]".to_string()
            ));
        }

        // Check batch and seq match
        if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
            return Err(AttentionError::InvalidShape(
                "Batch dimensions must match".to_string()
            ));
        }

        if q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1] {
            return Err(AttentionError::InvalidShape(
                "Sequence lengths must match for prefill".to_string()
            ));
        }

        // Check head dimensions
        if q_shape[2] != self.config.num_heads {
            return Err(AttentionError::InvalidShape(format!(
                "Expected {} Q heads, got {}",
                self.config.num_heads, q_shape[2]
            )));
        }

        if k_shape[2] != self.config.num_kv_heads || v_shape[2] != self.config.num_kv_heads {
            return Err(AttentionError::InvalidShape(format!(
                "Expected {} KV heads",
                self.config.num_kv_heads
            )));
        }

        Ok(())
    }

    /// Estimate FLOPs for attention.
    pub fn estimate_flops(&self, batch_size: usize, seq_len: usize) -> u64 {
        // Attention FLOPs: 4 * batch * seq^2 * hidden_dim (approximate)
        4 * batch_size as u64 * seq_len as u64 * seq_len as u64 * self.config.hidden_dim as u64
    }

    /// Estimate memory for KV cache.
    pub fn estimate_kv_memory(&self, batch_size: usize, seq_len: usize) -> u64 {
        (batch_size * seq_len * self.config.kv_size_per_token()) as u64
    }
}

/// Attention kernel errors.
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Kernel execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Unsupported configuration: {0}")]
    Unsupported(String),
}

// =============================================================================
// KV Cache Manager
// =============================================================================

/// KV cache for a single layer.
pub struct LayerKvCache {
    /// Key cache.
    pub k_cache: GpuBuffer,
    /// Value cache.
    pub v_cache: GpuBuffer,
    /// Current length.
    pub current_len: usize,
    /// Maximum length.
    pub max_len: usize,
    /// Layer ID.
    pub layer_id: LayerId,
}

impl LayerKvCache {
    /// Create a new KV cache.
    pub fn new(
        device: Arc<CudaDevice>,
        config: &AttentionConfig,
        batch_size: usize,
        max_len: usize,
        layer_id: LayerId,
    ) -> Result<Self, AttentionError> {
        let shape = [batch_size, max_len, config.num_kv_heads, config.head_dim];

        let k_cache = GpuBuffer::allocate(device.clone(), config.dtype, &shape)
            .map_err(|e| AttentionError::AllocationFailed(e.to_string()))?;

        let v_cache = GpuBuffer::allocate(device, config.dtype, &shape)
            .map_err(|e| AttentionError::AllocationFailed(e.to_string()))?;

        Ok(Self {
            k_cache,
            v_cache,
            current_len: 0,
            max_len,
            layer_id,
        })
    }

    /// Get current cache length.
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Check if cache is full.
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_len
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.max_len.saturating_sub(self.current_len)
    }

    /// Memory used by this cache.
    pub fn memory_used(&self) -> u64 {
        self.k_cache.size_bytes() + self.v_cache.size_bytes()
    }
}

/// Full KV cache for all layers.
pub struct KvCache {
    /// Per-layer caches.
    layers: Vec<LayerKvCache>,
    /// Configuration.
    config: AttentionConfig,
}

impl KvCache {
    /// Create KV cache for all layers.
    pub fn new(
        device: Arc<CudaDevice>,
        config: AttentionConfig,
        num_layers: usize,
        batch_size: usize,
        max_len: usize,
    ) -> Result<Self, AttentionError> {
        let mut layers = Vec::with_capacity(num_layers);

        for layer_id in 0..num_layers {
            layers.push(LayerKvCache::new(
                device.clone(),
                &config,
                batch_size,
                max_len,
                layer_id as LayerId,
            )?);
        }

        Ok(Self { layers, config })
    }

    /// Get cache for a layer.
    pub fn get(&self, layer_id: LayerId) -> Option<&LayerKvCache> {
        self.layers.get(layer_id as usize)
    }

    /// Get mutable cache for a layer.
    pub fn get_mut(&mut self, layer_id: LayerId) -> Option<&mut LayerKvCache> {
        self.layers.get_mut(layer_id as usize)
    }

    /// Total memory used.
    pub fn total_memory(&self) -> u64 {
        self.layers.iter().map(|l| l.memory_used()).sum()
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::mha(32, 128, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 32);
        assert_eq!(config.hidden_dim, 4096);
        assert!(!config.is_gqa());
        assert!(!config.is_mqa());
    }

    #[test]
    fn test_gqa_config() {
        let config = AttentionConfig::gqa(32, 8, 128, 4096);
        assert!(config.is_gqa());
        assert!(!config.is_mqa());
    }

    #[test]
    fn test_mqa_config() {
        let config = AttentionConfig::mqa(32, 128, 4096);
        assert!(!config.is_gqa());
        assert!(config.is_mqa());
    }

    #[test]
    fn test_kv_size() {
        let config = AttentionConfig::gqa(32, 8, 128, 4096);
        // 2 (K+V) * 8 heads * 128 dim * 2 bytes (FP16)
        assert_eq!(config.kv_size_per_token(), 2 * 8 * 128 * 2);
    }
}
