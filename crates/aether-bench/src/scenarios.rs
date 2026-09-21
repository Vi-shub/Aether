//! Benchmark scenarios for different use cases.

use serde::{Deserialize, Serialize};

/// Model configuration for benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name (for identification).
    pub name: String,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Intermediate dimension (FFN).
    pub intermediate_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Is this a MoE model?
    pub is_moe: bool,
    /// Number of experts (if MoE).
    pub num_experts: Option<usize>,
    /// Top-K experts (if MoE).
    pub top_k_experts: Option<usize>,
}

impl ModelConfig {
    /// LLaMA 7B configuration.
    pub fn llama_7b() -> Self {
        Self {
            name: "LLaMA-7B".to_string(),
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_dim: 11008,
            vocab_size: 32000,
            max_seq_len: 4096,
            is_moe: false,
            num_experts: None,
            top_k_experts: None,
        }
    }

    /// LLaMA 70B configuration.
    pub fn llama_70b() -> Self {
        Self {
            name: "LLaMA-70B".to_string(),
            num_layers: 80,
            hidden_dim: 8192,
            num_heads: 64,
            num_kv_heads: 8, // GQA
            head_dim: 128,
            intermediate_dim: 28672,
            vocab_size: 32000,
            max_seq_len: 4096,
            is_moe: false,
            num_experts: None,
            top_k_experts: None,
        }
    }

    /// Mixtral 8x7B configuration.
    pub fn mixtral_8x7b() -> Self {
        Self {
            name: "Mixtral-8x7B".to_string(),
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_dim: 14336,
            vocab_size: 32000,
            max_seq_len: 32768,
            is_moe: true,
            num_experts: Some(8),
            top_k_experts: Some(2),
        }
    }

    /// DeepSeek V2 (simplified).
    pub fn deepseek_v2() -> Self {
        Self {
            name: "DeepSeek-V2".to_string(),
            num_layers: 60,
            hidden_dim: 5120,
            num_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_dim: 12288,
            vocab_size: 100000,
            max_seq_len: 128000,
            is_moe: true,
            num_experts: Some(160),
            top_k_experts: Some(6),
        }
    }

    /// Estimate KV cache size per token (bytes).
    pub fn kv_bytes_per_token(&self) -> usize {
        // 2 (K+V) * num_kv_heads * head_dim * 2 (FP16)
        2 * self.num_kv_heads * self.head_dim * 2
    }

    /// Estimate total KV cache size (bytes) for given batch and seq length.
    pub fn kv_cache_size(&self, batch_size: usize, seq_len: usize) -> u64 {
        let per_token = self.kv_bytes_per_token();
        (self.num_layers * batch_size * seq_len * per_token) as u64
    }

    /// Estimate model weight size (bytes, FP16).
    pub fn model_size(&self) -> u64 {
        let embedding = self.vocab_size * self.hidden_dim * 2;

        let per_layer = if self.is_moe {
            // Attention + MoE
            let attention = 4 * self.hidden_dim * self.hidden_dim * 2; // QKV + O
            let expert_ffn = 3 * self.hidden_dim * self.intermediate_dim * 2; // gate, up, down
            let router = self.hidden_dim * self.num_experts.unwrap_or(8) * 2;
            attention + (expert_ffn * self.num_experts.unwrap_or(8)) + router
        } else {
            // Attention + FFN
            let attention = 4 * self.hidden_dim * self.hidden_dim * 2;
            let ffn = 3 * self.hidden_dim * self.intermediate_dim * 2;
            attention + ffn
        };

        (embedding + self.num_layers * per_layer) as u64
    }
}

/// Benchmark scenario types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkScenario {
    /// Memory transfer benchmarks.
    Transfer(TransferScenario),
    /// Decision engine benchmarks.
    Decision(DecisionScenario),
    /// Memory management benchmarks.
    Memory(MemoryScenario),
    /// End-to-end inference benchmarks.
    Inference(InferenceScenario),
}

/// Transfer benchmark scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferScenario {
    /// Sizes to benchmark (bytes).
    pub sizes: Vec<u64>,
    /// Number of iterations per size.
    pub iterations: usize,
    /// Warmup iterations.
    pub warmup: usize,
    /// Test H2D (host to device).
    pub test_h2d: bool,
    /// Test D2H (device to host).
    pub test_d2h: bool,
    /// Test D2D (device to device).
    pub test_d2d: bool,
    /// Test concurrent transfers.
    pub test_concurrent: bool,
}

impl Default for TransferScenario {
    fn default() -> Self {
        Self {
            sizes: vec![
                1024,                    // 1 KB
                1024 * 1024,             // 1 MB
                16 * 1024 * 1024,        // 16 MB
                128 * 1024 * 1024,       // 128 MB
                512 * 1024 * 1024,       // 512 MB
                1024 * 1024 * 1024,      // 1 GB
            ],
            iterations: 100,
            warmup: 10,
            test_h2d: true,
            test_d2h: true,
            test_d2d: true,
            test_concurrent: true,
        }
    }
}

/// Decision engine benchmark scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionScenario {
    /// Number of states.
    pub num_states: Vec<usize>,
    /// Number of operations in graph.
    pub num_operations: Vec<usize>,
    /// Memory pressure levels (0.0-1.0).
    pub memory_pressures: Vec<f64>,
    /// Iterations per configuration.
    pub iterations: usize,
}

impl Default for DecisionScenario {
    fn default() -> Self {
        Self {
            num_states: vec![100, 1000, 10000, 100000],
            num_operations: vec![32, 64, 128, 256],
            memory_pressures: vec![0.5, 0.75, 0.9, 0.95],
            iterations: 100,
        }
    }
}

/// Memory management benchmark scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryScenario {
    /// Pool size.
    pub pool_size: u64,
    /// Allocation sizes.
    pub allocation_sizes: Vec<u64>,
    /// Number of allocations.
    pub num_allocations: usize,
    /// Test fragmentation.
    pub test_fragmentation: bool,
}

impl Default for MemoryScenario {
    fn default() -> Self {
        Self {
            pool_size: 8 * 1024 * 1024 * 1024, // 8 GB
            allocation_sizes: vec![
                1024 * 1024,         // 1 MB
                16 * 1024 * 1024,    // 16 MB
                128 * 1024 * 1024,   // 128 MB
                512 * 1024 * 1024,   // 512 MB
            ],
            num_allocations: 1000,
            test_fragmentation: true,
        }
    }
}

/// End-to-end inference benchmark scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceScenario {
    /// Model configuration.
    pub model: ModelConfig,
    /// Batch sizes to test.
    pub batch_sizes: Vec<usize>,
    /// Sequence lengths to test (prefill).
    pub prefill_lengths: Vec<usize>,
    /// Decode steps.
    pub decode_steps: usize,
    /// GPU memory limit (for testing constrained scenarios).
    pub gpu_memory_limit: Option<u64>,
    /// Enable state offloading.
    pub enable_offload: bool,
    /// Enable recomputation.
    pub enable_recompute: bool,
}

impl InferenceScenario {
    /// Standard inference scenario for a model.
    pub fn standard(model: ModelConfig) -> Self {
        Self {
            model,
            batch_sizes: vec![1, 4, 8, 16],
            prefill_lengths: vec![128, 512, 2048],
            decode_steps: 128,
            gpu_memory_limit: None,
            enable_offload: false,
            enable_recompute: false,
        }
    }

    /// Memory-constrained scenario.
    pub fn memory_constrained(model: ModelConfig, gpu_memory_gb: u64) -> Self {
        Self {
            model,
            batch_sizes: vec![1, 2, 4],
            prefill_lengths: vec![128, 512],
            decode_steps: 64,
            gpu_memory_limit: Some(gpu_memory_gb * 1024 * 1024 * 1024),
            enable_offload: true,
            enable_recompute: true,
        }
    }

    /// Long-context scenario.
    pub fn long_context(model: ModelConfig) -> Self {
        Self {
            model,
            batch_sizes: vec![1, 2],
            prefill_lengths: vec![8192, 32768, 65536],
            decode_steps: 64,
            gpu_memory_limit: None,
            enable_offload: true,
            enable_recompute: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_7b_config() {
        let config = ModelConfig::llama_7b();
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.hidden_dim, 4096);
        assert!(!config.is_moe);
    }

    #[test]
    fn test_kv_cache_size() {
        let config = ModelConfig::llama_7b();
        // 32 layers * 1 batch * 2048 tokens * (2 * 32 * 128 * 2) bytes
        let expected = 32 * 1 * 2048 * (2 * 32 * 128 * 2);
        assert_eq!(config.kv_cache_size(1, 2048), expected as u64);
    }

    #[test]
    fn test_mixtral_moe() {
        let config = ModelConfig::mixtral_8x7b();
        assert!(config.is_moe);
        assert_eq!(config.num_experts, Some(8));
    }
}
