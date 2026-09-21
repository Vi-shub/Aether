//! Kernel executor - executes compute kernels with state awareness.

use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use aether_core::{ExecutionGraph, Operation, OperationId, StateId, Location};
use aether_cuda::{CudaDevice, CudaStream, StreamPool, GpuBuffer};
use aether_kernels::{
    AttentionKernel, AttentionConfig, KvCache,
    MlpKernel, MlpConfig,
    MoeKernel, MoeConfig, ExpertWeights,
    KernelRegistry,
};

use crate::{StateExecutor, TransferScheduler};

/// Kernel executor error.
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("Operation not found: {0}")]
    OperationNotFound(OperationId),
    #[error("State not available: {0}")]
    StateNotAvailable(StateId),
    #[error("Kernel not registered: {0}")]
    KernelNotRegistered(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Kernel execution result.
#[derive(Debug)]
pub struct KernelResult {
    /// Operation that was executed.
    pub operation_id: OperationId,
    /// Output state IDs.
    pub outputs: Vec<StateId>,
    /// Execution time in milliseconds.
    pub execution_time_ms: f64,
    /// Memory used during execution (peak).
    pub peak_memory_bytes: u64,
}

/// Kernel executor manages compute kernel execution.
pub struct KernelExecutor {
    /// CUDA device.
    device: Arc<CudaDevice>,
    /// Kernel registry.
    kernel_registry: Arc<KernelRegistry>,
    /// Stream pool.
    stream_pool: Arc<StreamPool>,
    /// State executor for data access.
    state_executor: Arc<StateExecutor>,
    /// Transfer scheduler for data movement.
    transfer_scheduler: Arc<TransferScheduler>,
    /// Statistics.
    stats: RwLock<ExecutorStats>,
}

/// Executor statistics.
#[derive(Debug, Default, Clone)]
pub struct ExecutorStats {
    pub operations_executed: u64,
    pub total_compute_time_ms: f64,
    pub total_wait_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl KernelExecutor {
    /// Create a new kernel executor.
    pub fn new(
        device: Arc<CudaDevice>,
        kernel_registry: Arc<KernelRegistry>,
        stream_pool: Arc<StreamPool>,
        state_executor: Arc<StateExecutor>,
        transfer_scheduler: Arc<TransferScheduler>,
    ) -> Self {
        Self {
            device,
            kernel_registry,
            stream_pool,
            state_executor,
            transfer_scheduler,
            stats: RwLock::new(ExecutorStats::default()),
        }
    }

    /// Execute a single operation.
    pub async fn execute_operation(
        &self,
        operation: &Operation,
    ) -> Result<KernelResult, KernelError> {
        let start = std::time::Instant::now();

        debug!(
            "Executing operation: id={}, type={:?}",
            operation.id, operation.operation_type
        );

        // Ensure all inputs are available on GPU
        self.ensure_inputs_available(&operation.inputs).await?;

        // Get a stream for execution
        let stream = self.stream_pool.acquire()
            .map_err(|e| KernelError::ExecutionFailed(e.to_string()))?;

        // Execute based on operation type
        let outputs = match operation.operation_type {
            aether_core::OperationType::Attention { layer_id } => {
                self.execute_attention(operation, layer_id, &stream).await?
            }
            aether_core::OperationType::FeedForward { layer_id } => {
                self.execute_mlp(operation, layer_id, &stream).await?
            }
            aether_core::OperationType::MoeRouting { layer_id } => {
                self.execute_moe_routing(operation, layer_id, &stream).await?
            }
            aether_core::OperationType::MoeExperts { layer_id } => {
                self.execute_moe_experts(operation, layer_id, &stream).await?
            }
            aether_core::OperationType::Embedding => {
                self.execute_embedding(operation, &stream).await?
            }
            aether_core::OperationType::Projection => {
                self.execute_projection(operation, &stream).await?
            }
            aether_core::OperationType::LayerNorm { layer_id } => {
                self.execute_norm(operation, layer_id, &stream).await?
            }
            aether_core::OperationType::Sampling => {
                self.execute_sampling(operation, &stream).await?
            }
            aether_core::OperationType::Custom(ref name) => {
                warn!("Custom operation not implemented: {}", name);
                Vec::new()
            }
        };

        // Synchronize stream
        stream.synchronize()
            .map_err(|e| KernelError::ExecutionFailed(e.to_string()))?;

        let execution_time = start.elapsed().as_secs_f64() * 1000.0;

        let mut stats = self.stats.write();
        stats.operations_executed += 1;
        stats.total_compute_time_ms += execution_time;

        Ok(KernelResult {
            operation_id: operation.id,
            outputs,
            execution_time_ms: execution_time,
            peak_memory_bytes: 0, // Would track actual memory
        })
    }

    /// Ensure all input states are available on GPU.
    async fn ensure_inputs_available(&self, inputs: &[StateId]) -> Result<(), KernelError> {
        for &state_id in inputs {
            let buffer = self.state_executor.get_buffer(state_id);

            match buffer {
                Some(crate::state_executor::PhysicalBuffer::Gpu(_)) => {
                    // Already on GPU
                    self.stats.write().cache_hits += 1;
                }
                Some(_) => {
                    // Need to transfer to GPU
                    self.stats.write().cache_misses += 1;

                    // Create transfer task
                    let task = crate::TransferTask::new(
                        state_id,
                        Location::Dram, // Would get actual location
                        Location::Hbm,
                        0, // Would get actual size
                    ).with_priority(crate::TransferPriority::Critical);

                    let task_id = self.transfer_scheduler.submit(task);
                    self.transfer_scheduler.wait_for(task_id).await;
                }
                None => {
                    return Err(KernelError::StateNotAvailable(state_id));
                }
            }
        }

        Ok(())
    }

    /// Execute attention operation.
    async fn execute_attention(
        &self,
        operation: &Operation,
        layer_id: usize,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        // Get the attention kernel
        let kernel_guard = self.kernel_registry.get_attention(layer_id)
            .ok_or_else(|| KernelError::KernelNotRegistered(
                format!("attention_{}", layer_id)
            ))?;

        // Execute with the kernel
        kernel_guard.with(|kernel| {
            debug!(
                "Attention layer {}: config={:?}",
                layer_id, kernel.config()
            );
            // Would call kernel.forward_prefill or forward_decode here
        });

        Ok(operation.outputs.clone())
    }

    /// Execute MLP operation.
    async fn execute_mlp(
        &self,
        operation: &Operation,
        layer_id: usize,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        debug!("MLP layer {}", layer_id);
        Ok(operation.outputs.clone())
    }

    /// Execute MoE routing operation.
    async fn execute_moe_routing(
        &self,
        operation: &Operation,
        layer_id: usize,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        debug!("MoE routing layer {}", layer_id);
        Ok(operation.outputs.clone())
    }

    /// Execute MoE experts operation.
    async fn execute_moe_experts(
        &self,
        operation: &Operation,
        layer_id: usize,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        debug!("MoE experts layer {}", layer_id);
        Ok(operation.outputs.clone())
    }

    /// Execute embedding operation.
    async fn execute_embedding(
        &self,
        operation: &Operation,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        debug!("Embedding lookup");
        Ok(operation.outputs.clone())
    }

    /// Execute projection operation.
    async fn execute_projection(
        &self,
        operation: &Operation,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        debug!("Projection");
        Ok(operation.outputs.clone())
    }

    /// Execute normalization operation.
    async fn execute_norm(
        &self,
        operation: &Operation,
        layer_id: usize,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        debug!("Norm layer {}", layer_id);
        Ok(operation.outputs.clone())
    }

    /// Execute sampling operation.
    async fn execute_sampling(
        &self,
        operation: &Operation,
        stream: &CudaStream,
    ) -> Result<Vec<StateId>, KernelError> {
        debug!("Sampling");
        Ok(operation.outputs.clone())
    }

    /// Execute operations in topological order.
    pub async fn execute_graph(
        &self,
        graph: &ExecutionGraph,
    ) -> Result<Vec<KernelResult>, KernelError> {
        let mut results = Vec::new();

        // Get operations in execution order
        let order = graph.topological_order();

        for op_id in order {
            if let Some(operation) = graph.get_operation(op_id) {
                let result = self.execute_operation(operation).await?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Get executor statistics.
    pub fn stats(&self) -> ExecutorStats {
        self.stats.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests would require more complex setup
}
