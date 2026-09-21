//! Mixture of Experts (MoE) kernel implementations.
//!
//! Supports:
//! - Top-K expert routing
//! - Expert prefetching integration
//! - Sparse expert execution

use std::sync::Arc;
use std::collections::HashMap;
use aether_core::{DType, StateId};
use aether_cuda::{CudaDevice, GpuBuffer, CudaStream};
use super::mlp::{MlpConfig, Activation};

/// MoE configuration.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of experts.
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub top_k: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Expert intermediate dimension.
    pub expert_intermediate_dim: usize,
    /// Data type.
    pub dtype: DType,
    /// Activation function.
    pub activation: Activation,
    /// Whether experts use gated activation.
    pub gated: bool,
    /// Router jitter for training.
    pub router_jitter: f32,
    /// Capacity factor.
    pub capacity_factor: f32,
}

impl MoeConfig {
    /// Create Mixtral-style MoE (8 experts, top-2).
    pub fn mixtral(hidden_dim: usize, expert_intermediate_dim: usize) -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            hidden_dim,
            expert_intermediate_dim,
            dtype: DType::Float16,
            activation: Activation::Silu,
            gated: true,
            router_jitter: 0.0,
            capacity_factor: 1.0,
        }
    }

    /// Create DeepSeek-style MoE (many experts, top-6).
    pub fn deepseek(
        num_experts: usize,
        top_k: usize,
        hidden_dim: usize,
        expert_intermediate_dim: usize,
    ) -> Self {
        Self {
            num_experts,
            top_k,
            hidden_dim,
            expert_intermediate_dim,
            dtype: DType::Float16,
            activation: Activation::Silu,
            gated: true,
            router_jitter: 0.0,
            capacity_factor: 1.0,
        }
    }

    /// Get per-expert weight size.
    pub fn expert_weight_size(&self) -> usize {
        let mlp_config = MlpConfig {
            hidden_dim: self.hidden_dim,
            intermediate_dim: self.expert_intermediate_dim,
            dtype: self.dtype,
            activation: self.activation,
            gated: self.gated,
        };
        mlp_config.total_weight_size()
    }

    /// Total weight size for all experts.
    pub fn total_weight_size(&self) -> usize {
        self.expert_weight_size() * self.num_experts
    }

    /// Router weight size.
    pub fn router_weight_size(&self) -> usize {
        self.hidden_dim * self.num_experts * self.dtype.size_bytes()
    }
}

/// Expert selection result.
#[derive(Debug, Clone)]
pub struct ExpertSelection {
    /// Selected expert indices per token [batch * seq, top_k].
    pub expert_indices: Vec<Vec<usize>>,
    /// Routing weights per token [batch * seq, top_k].
    pub routing_weights: Vec<Vec<f32>>,
    /// Number of tokens assigned to each expert.
    pub expert_counts: HashMap<usize, usize>,
}

impl ExpertSelection {
    /// Get experts needed for this selection.
    pub fn needed_experts(&self) -> Vec<usize> {
        self.expert_counts.keys().copied().collect()
    }

    /// Check if an expert is needed.
    pub fn needs_expert(&self, expert_id: usize) -> bool {
        self.expert_counts.contains_key(&expert_id)
    }

    /// Total tokens.
    pub fn total_tokens(&self) -> usize {
        self.expert_indices.len()
    }
}

/// MoE kernel.
pub struct MoeKernel {
    config: MoeConfig,
    device: Arc<CudaDevice>,
    /// Expert weight buffer map (expert_id -> state_id).
    expert_states: HashMap<usize, StateId>,
}

impl MoeKernel {
    /// Create new MoE kernel.
    pub fn new(device: Arc<CudaDevice>, config: MoeConfig) -> Self {
        Self {
            config,
            device,
            expert_states: HashMap::new(),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &MoeConfig {
        &self.config
    }

    /// Register expert weights with state manager.
    pub fn register_expert(&mut self, expert_id: usize, state_id: StateId) {
        self.expert_states.insert(expert_id, state_id);
    }

    /// Get state ID for an expert.
    pub fn expert_state(&self, expert_id: usize) -> Option<StateId> {
        self.expert_states.get(&expert_id).copied()
    }

    /// Compute routing (expert selection).
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq_len, hidden_dim]
    /// * `router_weights` - Router weight [hidden_dim, num_experts]
    /// * `stream` - CUDA stream
    pub fn compute_routing(
        &self,
        hidden_states: &GpuBuffer,
        router_weights: &GpuBuffer,
        stream: Option<&CudaStream>,
    ) -> Result<ExpertSelection, MoeError> {
        let shape = hidden_states.shape();
        if shape.len() != 3 {
            return Err(MoeError::InvalidShape(
                "Expected 3D input [batch, seq, hidden]".to_string()
            ));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let total_tokens = batch_size * seq_len;

        // In production, this would:
        // 1. GEMM: hidden_states @ router_weights -> router_logits [batch*seq, num_experts]
        // 2. TopK selection
        // 3. Softmax over selected experts

        // Mock implementation: simulate routing
        let mut expert_indices = Vec::with_capacity(total_tokens);
        let mut routing_weights = Vec::with_capacity(total_tokens);
        let mut expert_counts: HashMap<usize, usize> = HashMap::new();

        for token_idx in 0..total_tokens {
            // Simulate top-k selection (would be actual computation)
            let mut selected: Vec<usize> = (0..self.config.top_k)
                .map(|k| (token_idx + k) % self.config.num_experts)
                .collect();
            selected.sort();
            selected.dedup();
            while selected.len() < self.config.top_k {
                selected.push(selected[0]);
            }

            let weights: Vec<f32> = (0..self.config.top_k)
                .map(|_| 1.0 / self.config.top_k as f32)
                .collect();

            for &expert_id in &selected {
                *expert_counts.entry(expert_id).or_insert(0) += 1;
            }

            expert_indices.push(selected);
            routing_weights.push(weights);
        }

        Ok(ExpertSelection {
            expert_indices,
            routing_weights,
            expert_counts,
        })
    }

    /// Execute MoE forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor
    /// * `selection` - Expert selection from routing
    /// * `expert_weights` - Map of expert_id -> (up_weight, gate_weight, down_weight)
    /// * `stream` - CUDA stream
    pub fn forward(
        &self,
        hidden_states: &GpuBuffer,
        selection: &ExpertSelection,
        expert_weights: &HashMap<usize, ExpertWeights>,
        stream: Option<&CudaStream>,
    ) -> Result<GpuBuffer, MoeError> {
        // Validate all needed experts are available
        for expert_id in selection.needed_experts() {
            if !expert_weights.contains_key(&expert_id) {
                return Err(MoeError::ExpertNotLoaded(expert_id));
            }
        }

        // Allocate output
        let output = GpuBuffer::allocate(
            self.device.clone(),
            self.config.dtype,
            hidden_states.shape(),
        ).map_err(|e| MoeError::AllocationFailed(e.to_string()))?;

        // In production, this would:
        // 1. Group tokens by assigned expert
        // 2. For each expert with assigned tokens:
        //    a. Gather tokens -> expert_input
        //    b. Expert MLP forward
        //    c. Scale by routing weight
        //    d. Scatter-add to output
        // 3. Can be parallelized across experts

        tracing::debug!(
            "MoE forward: {} tokens, {} experts active",
            selection.total_tokens(),
            selection.needed_experts().len()
        );

        Ok(output)
    }

    /// Estimate FLOPs.
    pub fn estimate_flops(&self, batch_size: usize, seq_len: usize) -> u64 {
        let tokens = batch_size as u64 * seq_len as u64;
        let hidden = self.config.hidden_dim as u64;
        let intermediate = self.config.expert_intermediate_dim as u64;
        let top_k = self.config.top_k as u64;

        // Router: tokens * hidden * num_experts * 2
        let router_flops = tokens * hidden * self.config.num_experts as u64 * 2;

        // Expert MLPs: tokens * top_k * MLP_flops
        let multiplier = if self.config.gated { 3 } else { 2 };
        let expert_flops = tokens * top_k * hidden * intermediate * 2 * multiplier;

        router_flops + expert_flops
    }

    /// Get experts that should be prefetched based on prediction.
    pub fn predict_experts(&self, _context: &[usize]) -> Vec<usize> {
        // In production, this would use:
        // - Historical expert usage patterns
        // - Speculative routing based on input features
        // - Model-specific heuristics

        // Mock: return most common experts
        (0..self.config.top_k.min(self.config.num_experts)).collect()
    }
}

/// Expert weights.
#[derive(Debug)]
pub struct ExpertWeights {
    /// Up projection weight.
    pub up_weight: GpuBuffer,
    /// Gate weight (for gated MLP).
    pub gate_weight: Option<GpuBuffer>,
    /// Down projection weight.
    pub down_weight: GpuBuffer,
}

/// MoE errors.
#[derive(Debug, thiserror::Error)]
pub enum MoeError {
    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),
    #[error("Expert {0} not loaded")]
    ExpertNotLoaded(usize),
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Kernel execution failed: {0}")]
    ExecutionFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixtral_config() {
        let config = MoeConfig::mixtral(4096, 14336);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);
    }

    #[test]
    fn test_deepseek_config() {
        let config = MoeConfig::deepseek(160, 6, 7168, 2048);
        assert_eq!(config.num_experts, 160);
        assert_eq!(config.top_k, 6);
    }

    #[test]
    fn test_expert_weight_size() {
        let config = MoeConfig::mixtral(4096, 14336);
        let expert_size = config.expert_weight_size();
        // Should be significant (gated MLP weights)
        assert!(expert_size > 0);
    }
}
