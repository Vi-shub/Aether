//! Unified runtime - the main Aether runtime integrating all components.

use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info, warn, error};

use aether_core::{
    CostModel, DecisionEngine, ExecutionGraph, ExecutionPlan, ExecutionState,
    HardwareConfig, Location, Operation, OperationType, StateId, StateManager,
    StateType, DType,
};
use aether_cuda::{CudaDevice, MemoryPool, StreamPool, TransferEngine};
use aether_kernels::KernelRegistry;

use crate::{
    KernelExecutor, MetricsCollector, RuntimeMetrics, StateExecutor,
    TransferScheduler, TransferPriority, TransferTask,
};

/// Runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Hardware configuration.
    pub hardware: HardwareConfig,
    /// Maximum concurrent transfers.
    pub max_concurrent_transfers: usize,
    /// Number of CUDA streams.
    pub num_streams: usize,
    /// GPU memory pool size (bytes).
    pub gpu_pool_size: u64,
    /// Enable speculative prefetching.
    pub enable_prefetch: bool,
    /// Prefetch lookahead (number of operations).
    pub prefetch_lookahead: usize,
    /// Enable recomputation.
    pub enable_recompute: bool,
    /// Memory pressure threshold (0.0-1.0).
    pub memory_pressure_threshold: f64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            hardware: HardwareConfig::h100(),
            max_concurrent_transfers: 4,
            num_streams: 8,
            gpu_pool_size: 16 * 1024 * 1024 * 1024, // 16 GB
            enable_prefetch: true,
            prefetch_lookahead: 4,
            enable_recompute: true,
            memory_pressure_threshold: 0.85,
        }
    }
}

impl RuntimeConfig {
    /// Create config for A100 GPU.
    pub fn a100() -> Self {
        Self {
            hardware: HardwareConfig::a100(),
            gpu_pool_size: 40 * 1024 * 1024 * 1024, // 40 GB
            ..Default::default()
        }
    }

    /// Create config for H100 GPU.
    pub fn h100() -> Self {
        Self {
            hardware: HardwareConfig::h100(),
            gpu_pool_size: 80 * 1024 * 1024 * 1024, // 80 GB
            ..Default::default()
        }
    }

    /// Create config for consumer GPU (RTX 4090).
    pub fn rtx_4090() -> Self {
        Self {
            hardware: HardwareConfig::rtx_4090(),
            gpu_pool_size: 24 * 1024 * 1024 * 1024, // 24 GB
            ..Default::default()
        }
    }

    /// Create memory-constrained config.
    pub fn memory_constrained(gpu_memory_gb: u64) -> Self {
        Self {
            gpu_pool_size: gpu_memory_gb * 1024 * 1024 * 1024,
            enable_recompute: true,
            memory_pressure_threshold: 0.75,
            ..Default::default()
        }
    }
}

/// Unified Aether runtime.
pub struct UnifiedRuntime {
    /// Configuration.
    config: RuntimeConfig,
    /// CUDA device.
    device: Arc<CudaDevice>,
    /// State manager.
    state_manager: Arc<StateManager>,
    /// Decision engine.
    decision_engine: Arc<DecisionEngine>,
    /// Cost model.
    cost_model: Arc<CostModel>,
    /// State executor.
    state_executor: Arc<StateExecutor>,
    /// Transfer scheduler.
    transfer_scheduler: Arc<TransferScheduler>,
    /// Kernel executor.
    kernel_executor: Arc<KernelExecutor>,
    /// Kernel registry.
    kernel_registry: Arc<KernelRegistry>,
    /// Memory pool.
    memory_pool: Arc<MemoryPool>,
    /// Stream pool.
    stream_pool: Arc<StreamPool>,
    /// Transfer engine.
    transfer_engine: Arc<TransferEngine>,
    /// Metrics collector.
    metrics: Arc<MetricsCollector>,
    /// Current execution graph (if any).
    current_graph: RwLock<Option<Arc<ExecutionGraph>>>,
}

impl UnifiedRuntime {
    /// Create a new unified runtime.
    pub fn new(config: RuntimeConfig) -> Result<Self, RuntimeError> {
        info!("Initializing Aether runtime...");

        // Initialize CUDA device
        let device = Arc::new(CudaDevice::new(0)
            .map_err(|e| RuntimeError::Initialization(e.to_string()))?);

        info!("CUDA device initialized: {}", device.name());

        // Create components
        let state_manager = Arc::new(StateManager::new(config.hardware.clone()));
        let cost_model = Arc::new(CostModel::new(config.hardware.clone()));
        let decision_engine = Arc::new(DecisionEngine::new(
            cost_model.clone(),
            config.hardware.clone(),
        ));

        let memory_pool = Arc::new(MemoryPool::new(device.clone(), config.gpu_pool_size));
        let stream_pool = Arc::new(StreamPool::new(device.clone(), config.num_streams)
            .map_err(|e| RuntimeError::Initialization(e.to_string()))?);
        let transfer_engine = Arc::new(TransferEngine::new(device.clone())
            .map_err(|e| RuntimeError::Initialization(e.to_string()))?);

        let kernel_registry = Arc::new(KernelRegistry::new(device.clone()));

        let state_executor = Arc::new(StateExecutor::new(
            state_manager.clone(),
            device.clone(),
            transfer_engine.clone(),
            memory_pool.clone(),
            stream_pool.clone(),
        ));

        let transfer_scheduler = Arc::new(TransferScheduler::new(
            device.clone(),
            transfer_engine.clone(),
            stream_pool.clone(),
            config.max_concurrent_transfers,
        ));

        let kernel_executor = Arc::new(KernelExecutor::new(
            device.clone(),
            kernel_registry.clone(),
            stream_pool.clone(),
            state_executor.clone(),
            transfer_scheduler.clone(),
        ));

        let metrics = Arc::new(MetricsCollector::new());

        info!("Aether runtime initialized successfully");

        Ok(Self {
            config,
            device,
            state_manager,
            decision_engine,
            cost_model,
            state_executor,
            transfer_scheduler,
            kernel_executor,
            kernel_registry,
            memory_pool,
            stream_pool,
            transfer_engine,
            metrics,
            current_graph: RwLock::new(None),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get the state manager.
    pub fn state_manager(&self) -> &Arc<StateManager> {
        &self.state_manager
    }

    /// Get the kernel registry.
    pub fn kernel_registry(&self) -> &Arc<KernelRegistry> {
        &self.kernel_registry
    }

    /// Register an execution graph.
    pub fn set_execution_graph(&self, graph: ExecutionGraph) {
        *self.current_graph.write() = Some(Arc::new(graph));
    }

    /// Plan execution for the current graph.
    pub fn plan(&self) -> Result<ExecutionPlan, RuntimeError> {
        let graph = self.current_graph.read();
        let graph = graph.as_ref()
            .ok_or(RuntimeError::NoGraph)?;

        let states: Vec<_> = self.state_manager.all_states();

        let plan = self.decision_engine.plan(&states, graph);
        info!("Created execution plan with {} decisions", plan.len());

        Ok(plan)
    }

    /// Execute the current graph.
    pub async fn execute(&self) -> Result<(), RuntimeError> {
        // Plan
        let plan = self.plan()?;

        // Execute state decisions
        self.state_executor.execute_plan(&plan).await
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // Execute kernels
        if let Some(graph) = self.current_graph.read().as_ref() {
            self.kernel_executor.execute_graph(graph).await
                .map_err(|e| RuntimeError::Execution(e.to_string()))?;
        }

        Ok(())
    }

    /// Execute a single forward pass (prefill or decode).
    pub async fn forward(
        &self,
        input_ids: &[u32],
        is_prefill: bool,
    ) -> Result<Vec<f32>, RuntimeError> {
        let start = std::time::Instant::now();

        // In production:
        // 1. Plan state movement based on inputs
        // 2. Execute transfers
        // 3. Execute compute kernels
        // 4. Return logits

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "Forward pass: {} tokens, prefill={}, time={:.2}ms",
            input_ids.len(), is_prefill, elapsed
        );

        // Mock output
        Ok(vec![0.0; input_ids.len() * 1000]) // Mock vocab=1000
    }

    /// Check memory pressure and trigger eviction if needed.
    pub fn check_memory_pressure(&self) -> bool {
        let stats = self.memory_pool.stats();
        let utilization = stats.bytes_allocated as f64 / self.config.gpu_pool_size as f64;

        if utilization > self.config.memory_pressure_threshold {
            warn!(
                "High memory pressure: {:.1}% utilized",
                utilization * 100.0
            );
            true
        } else {
            false
        }
    }

    /// Prefetch states for upcoming operations.
    pub async fn prefetch_for_operations(&self, operations: &[OperationType]) {
        if !self.config.enable_prefetch {
            return;
        }

        // Would analyze operations to determine required states
        // and initiate prefetch transfers
        debug!(
            "Prefetching for {} upcoming operations",
            operations.len()
        );
    }

    /// Get runtime metrics.
    pub fn metrics(&self) -> RuntimeMetrics {
        self.metrics.collect(&self)
    }

    /// Synchronize all pending operations.
    pub async fn synchronize(&self) {
        self.transfer_scheduler.wait_all().await;
        self.device.synchronize().ok();
    }

    /// Shutdown the runtime.
    pub async fn shutdown(&self) {
        info!("Shutting down Aether runtime...");
        self.synchronize().await;
        info!("Aether runtime shut down");
    }
}

/// Runtime errors.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Initialization failed: {0}")]
    Initialization(String),
    #[error("No execution graph set")]
    NoGraph,
    #[error("Execution failed: {0}")]
    Execution(String),
    #[error("Memory allocation failed: {0}")]
    OutOfMemory(String),
}

// =============================================================================
// Builder pattern for runtime
// =============================================================================

/// Builder for UnifiedRuntime.
pub struct RuntimeBuilder {
    config: RuntimeConfig,
    device_id: i32,
}

impl RuntimeBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
            device_id: 0,
        }
    }

    /// Set hardware config.
    pub fn hardware(mut self, hardware: HardwareConfig) -> Self {
        self.config.hardware = hardware;
        self
    }

    /// Set GPU device ID.
    pub fn device(mut self, device_id: i32) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set GPU memory pool size.
    pub fn gpu_memory(mut self, bytes: u64) -> Self {
        self.config.gpu_pool_size = bytes;
        self
    }

    /// Set number of CUDA streams.
    pub fn streams(mut self, num: usize) -> Self {
        self.config.num_streams = num;
        self
    }

    /// Enable/disable prefetching.
    pub fn prefetch(mut self, enable: bool) -> Self {
        self.config.enable_prefetch = enable;
        self
    }

    /// Enable/disable recomputation.
    pub fn recompute(mut self, enable: bool) -> Self {
        self.config.enable_recompute = enable;
        self
    }

    /// Build the runtime.
    pub fn build(self) -> Result<UnifiedRuntime, RuntimeError> {
        UnifiedRuntime::new(self.config)
    }
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = RuntimeConfig::default();
        assert!(config.enable_prefetch);
        assert!(config.enable_recompute);
    }

    #[test]
    fn test_config_profiles() {
        let a100 = RuntimeConfig::a100();
        assert_eq!(a100.gpu_pool_size, 40 * 1024 * 1024 * 1024);

        let h100 = RuntimeConfig::h100();
        assert_eq!(h100.gpu_pool_size, 80 * 1024 * 1024 * 1024);
    }
}
