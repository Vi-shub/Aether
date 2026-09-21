//! Sampling kernel implementations.
//!
//! Supports:
//! - Greedy decoding
//! - Top-K sampling
//! - Top-P (nucleus) sampling
//! - Temperature scaling
//! - Beam search

use std::sync::Arc;
use aether_core::DType;
use aether_cuda::{CudaDevice, GpuBuffer, CudaStream};

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for scaling logits.
    pub temperature: f32,
    /// Top-K value (0 = disabled).
    pub top_k: usize,
    /// Top-P (nucleus) value (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// Presence penalty.
    pub presence_penalty: f32,
    /// Frequency penalty.
    pub frequency_penalty: f32,
    /// Minimum tokens to generate.
    pub min_tokens: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Stop sequences.
    pub stop_sequences: Vec<Vec<u32>>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_tokens: 1,
            max_tokens: 2048,
            stop_sequences: Vec::new(),
        }
    }
}

impl SamplingConfig {
    /// Create greedy decoding config.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        }
    }

    /// Create sampling config with temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Create top-K sampling config.
    pub fn top_k(k: usize, temperature: f32) -> Self {
        Self {
            temperature,
            top_k: k,
            ..Default::default()
        }
    }

    /// Create top-P (nucleus) sampling config.
    pub fn top_p(p: f32, temperature: f32) -> Self {
        Self {
            temperature,
            top_p: p,
            ..Default::default()
        }
    }

    /// Check if this is greedy decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

/// Sampling kernel.
pub struct SamplingKernel {
    config: SamplingConfig,
    vocab_size: usize,
    dtype: DType,
    device: Arc<CudaDevice>,
}

impl SamplingKernel {
    /// Create new sampling kernel.
    pub fn new(
        device: Arc<CudaDevice>,
        vocab_size: usize,
        config: SamplingConfig,
        dtype: DType,
    ) -> Self {
        Self {
            config,
            vocab_size,
            dtype,
            device,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: SamplingConfig) {
        self.config = config;
    }

    /// Sample from logits.
    ///
    /// # Arguments
    /// * `logits` - Logits tensor [batch, vocab_size]
    /// * `stream` - CUDA stream
    ///
    /// # Returns
    /// Token IDs [batch]
    pub fn sample(
        &self,
        logits: &GpuBuffer,
        stream: Option<&CudaStream>,
    ) -> Result<Vec<u32>, SamplingError> {
        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(SamplingError::InvalidShape(
                "Expected 2D logits [batch, vocab]".to_string()
            ));
        }

        let batch_size = shape[0];
        let vocab = shape[1];

        if vocab != self.vocab_size {
            return Err(SamplingError::InvalidShape(format!(
                "Expected vocab size {}, got {}",
                self.vocab_size, vocab
            )));
        }

        // In production, this would:
        // 1. Apply temperature: logits = logits / temperature
        // 2. Apply top-k masking
        // 3. Apply top-p masking
        // 4. Softmax
        // 5. Sample from distribution (or argmax for greedy)

        // Mock implementation
        let tokens: Vec<u32> = (0..batch_size).map(|i| (i % vocab) as u32).collect();

        tracing::debug!(
            "Sampling: batch={}, greedy={}, temp={}",
            batch_size,
            self.config.is_greedy(),
            self.config.temperature
        );

        Ok(tokens)
    }

    /// Sample with repetition penalty.
    pub fn sample_with_history(
        &self,
        logits: &GpuBuffer,
        token_history: &[Vec<u32>],
        stream: Option<&CudaStream>,
    ) -> Result<Vec<u32>, SamplingError> {
        // Apply repetition penalty based on token history
        // Then sample normally
        self.sample(logits, stream)
    }

    /// Beam search step.
    pub fn beam_step(
        &self,
        logits: &GpuBuffer,
        beam_scores: &[f32],
        num_beams: usize,
        stream: Option<&CudaStream>,
    ) -> Result<BeamSearchOutput, SamplingError> {
        let shape = logits.shape();
        let batch_beam = shape[0];

        if batch_beam % num_beams != 0 {
            return Err(SamplingError::InvalidShape(
                "Batch size must be divisible by num_beams".to_string()
            ));
        }

        let batch_size = batch_beam / num_beams;

        // In production:
        // 1. Compute log_probs = log_softmax(logits)
        // 2. Add beam scores: scores = beam_scores + log_probs
        // 3. For each batch, select top num_beams from all beam candidates

        // Mock output
        let tokens: Vec<u32> = vec![0; batch_beam];
        let scores: Vec<f32> = vec![0.0; batch_beam];
        let beam_indices: Vec<usize> = (0..batch_beam).map(|i| i % num_beams).collect();

        Ok(BeamSearchOutput {
            tokens,
            scores,
            beam_indices,
        })
    }
}

/// Beam search output.
#[derive(Debug)]
pub struct BeamSearchOutput {
    /// Selected token IDs.
    pub tokens: Vec<u32>,
    /// Updated beam scores.
    pub scores: Vec<f32>,
    /// Indices of beams that were extended.
    pub beam_indices: Vec<usize>,
}

/// Sampling errors.
#[derive(Debug, thiserror::Error)]
pub enum SamplingError {
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
    fn test_greedy_config() {
        let config = SamplingConfig::greedy();
        assert!(config.is_greedy());
    }

    #[test]
    fn test_top_k_config() {
        let config = SamplingConfig::top_k(50, 0.8);
        assert!(!config.is_greedy());
        assert_eq!(config.top_k, 50);
        assert_eq!(config.temperature, 0.8);
    }

    #[test]
    fn test_top_p_config() {
        let config = SamplingConfig::top_p(0.9, 0.7);
        assert!(!config.is_greedy());
        assert_eq!(config.top_p, 0.9);
    }
}
