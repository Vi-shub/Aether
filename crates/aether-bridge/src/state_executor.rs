//! State executor - translates ExecutionPlan decisions into physical operations.

use std::sync::Arc;
use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use aether_core::{
    Action, ExecutionPlan, ExecutionState, Location, StateDecision, StateId, StateManager,
};
use aether_cuda::{
    CudaDevice, CudaStream, GpuBuffer, CpuBuffer, TransferEngine, MemoryPool,
    BufferRegistry, StreamPool,
};

use crate::TransferScheduler;

/// State executor error.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("State not found: {0}")]
    StateNotFound(StateId),
    #[error("Buffer not found for state: {0}")]
    BufferNotFound(StateId),
    #[error("Transfer failed: {0}")]
    TransferFailed(String),
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
}

/// Physical buffer location.
#[derive(Debug, Clone)]
pub enum PhysicalBuffer {
    Gpu(GpuBuffer),
    Cpu(CpuBuffer),
    /// Buffer is in NVMe - path to backing file
    Nvme(String),
}

impl PhysicalBuffer {
    /// Get size in bytes.
    pub fn size_bytes(&self) -> u64 {
        match self {
            PhysicalBuffer::Gpu(buf) => buf.size_bytes(),
            PhysicalBuffer::Cpu(buf) => buf.size_bytes(),
            PhysicalBuffer::Nvme(_) => 0, // Would need file size lookup
        }
    }
}

/// State executor translates logical decisions into physical operations.
pub struct StateExecutor {
    /// Core state manager (logical state tracking).
    state_manager: Arc<StateManager>,
    /// Physical buffer storage.
    buffers: DashMap<StateId, PhysicalBuffer>,
    /// CUDA device.
    device: Arc<CudaDevice>,
    /// Transfer engine.
    transfer_engine: Arc<TransferEngine>,
    /// Memory pool for GPU allocations.
    memory_pool: Arc<MemoryPool>,
    /// Stream pool.
    stream_pool: Arc<StreamPool>,
    /// Buffer registry for tracking.
    buffer_registry: Arc<BufferRegistry>,
    /// Execution statistics.
    stats: RwLock<ExecutorStats>,
}

/// Executor statistics.
#[derive(Debug, Default, Clone)]
pub struct ExecutorStats {
    pub actions_executed: u64,
    pub moves_completed: u64,
    pub recomputes_completed: u64,
    pub evictions_completed: u64,
    pub bytes_transferred: u64,
    pub errors: u64,
}

impl StateExecutor {
    /// Create a new state executor.
    pub fn new(
        state_manager: Arc<StateManager>,
        device: Arc<CudaDevice>,
        transfer_engine: Arc<TransferEngine>,
        memory_pool: Arc<MemoryPool>,
        stream_pool: Arc<StreamPool>,
    ) -> Self {
        Self {
            state_manager,
            buffers: DashMap::new(),
            device: device.clone(),
            transfer_engine,
            memory_pool,
            stream_pool,
            buffer_registry: Arc::new(BufferRegistry::new()),
            stats: RwLock::new(ExecutorStats::default()),
        }
    }

    /// Execute an entire execution plan.
    pub async fn execute_plan(&self, plan: &ExecutionPlan) -> Result<(), ExecutorError> {
        info!(
            "Executing plan with {} decisions, estimated time: {:.2}ms",
            plan.len(),
            plan.total_time_ms()
        );

        for decision in plan.iter() {
            self.execute_decision(decision).await?;
        }

        Ok(())
    }

    /// Execute a single state decision.
    pub async fn execute_decision(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        debug!(
            "Executing decision: state={}, action={:?}, target={:?}",
            decision.state_id, decision.action, decision.target_location
        );

        let result = match decision.action {
            Action::Keep => self.execute_keep(decision).await,
            Action::Move => self.execute_move(decision).await,
            Action::Prefetch => self.execute_prefetch(decision).await,
            Action::Replicate => self.execute_replicate(decision).await,
            Action::Recompute => self.execute_recompute(decision).await,
            Action::Compress => self.execute_compress(decision).await,
            Action::Approximate => self.execute_approximate(decision).await,
            Action::Evict => self.execute_evict(decision).await,
            Action::Delay => self.execute_delay(decision).await,
        };

        let mut stats = self.stats.write();
        stats.actions_executed += 1;

        if result.is_err() {
            stats.errors += 1;
        }

        result
    }

    /// Execute KEEP action (no-op, state stays where it is).
    async fn execute_keep(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        debug!("KEEP: state {} stays in place", decision.state_id);
        Ok(())
    }

    /// Execute MOVE action.
    async fn execute_move(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        let state_id = decision.state_id;
        let target = decision.target_location.ok_or_else(|| {
            ExecutorError::InvalidAction("MOVE requires target_location".to_string())
        })?;

        // Get current state info
        let state = self.state_manager.get_state(state_id)
            .ok_or(ExecutorError::StateNotFound(state_id))?;
        let current_location = state.location;

        debug!(
            "MOVE: state {} from {:?} to {:?}",
            state_id, current_location, target
        );

        // Get source buffer
        let source_buffer = self.buffers.get(&state_id)
            .ok_or(ExecutorError::BufferNotFound(state_id))?;

        // Perform the move based on source/target
        match (current_location, target) {
            (Location::Hbm, Location::Dram) => {
                self.move_gpu_to_cpu(state_id, &source_buffer).await?;
            }
            (Location::Dram, Location::Hbm) => {
                self.move_cpu_to_gpu(state_id, &source_buffer).await?;
            }
            (Location::Hbm, Location::Nvme) => {
                self.move_gpu_to_nvme(state_id, &source_buffer).await?;
            }
            (Location::Nvme, Location::Hbm) => {
                self.move_nvme_to_gpu(state_id, &source_buffer).await?;
            }
            (Location::Dram, Location::Nvme) => {
                self.move_cpu_to_nvme(state_id, &source_buffer).await?;
            }
            (Location::Nvme, Location::Dram) => {
                self.move_nvme_to_cpu(state_id, &source_buffer).await?;
            }
            _ => {
                warn!("Unsupported move path: {:?} -> {:?}", current_location, target);
            }
        }

        // Update state manager
        self.state_manager.update_location(state_id, target);

        let mut stats = self.stats.write();
        stats.moves_completed += 1;
        stats.bytes_transferred += state.size_bytes;

        Ok(())
    }

    /// Move data from GPU to CPU.
    async fn move_gpu_to_cpu(
        &self,
        state_id: StateId,
        source: &PhysicalBuffer,
    ) -> Result<(), ExecutorError> {
        let gpu_buffer = match source.value() {
            PhysicalBuffer::Gpu(buf) => buf,
            _ => return Err(ExecutorError::InvalidAction("Source not on GPU".to_string())),
        };

        // Allocate CPU buffer
        let cpu_buffer = CpuBuffer::allocate(
            gpu_buffer.dtype(),
            gpu_buffer.shape(),
        ).map_err(|e| ExecutorError::CudaError(e.to_string()))?;

        // Perform transfer
        let stream = self.stream_pool.acquire()
            .map_err(|e| ExecutorError::CudaError(e.to_string()))?;

        // Note: In production, this would use the transfer engine's async copy
        // and properly wait for completion

        // Update buffer storage
        drop(source);
        self.buffers.insert(state_id, PhysicalBuffer::Cpu(cpu_buffer));

        Ok(())
    }

    /// Move data from CPU to GPU.
    async fn move_cpu_to_gpu(
        &self,
        state_id: StateId,
        source: &PhysicalBuffer,
    ) -> Result<(), ExecutorError> {
        let cpu_buffer = match source.value() {
            PhysicalBuffer::Cpu(buf) => buf,
            _ => return Err(ExecutorError::InvalidAction("Source not on CPU".to_string())),
        };

        // Allocate GPU buffer from pool
        let gpu_buffer = GpuBuffer::allocate(
            self.device.clone(),
            cpu_buffer.dtype(),
            cpu_buffer.shape(),
        ).map_err(|e| ExecutorError::CudaError(e.to_string()))?;

        // Perform transfer
        let stream = self.stream_pool.acquire()
            .map_err(|e| ExecutorError::CudaError(e.to_string()))?;

        // Update buffer storage
        drop(source);
        self.buffers.insert(state_id, PhysicalBuffer::Gpu(gpu_buffer));

        Ok(())
    }

    /// Move data from GPU to NVMe.
    async fn move_gpu_to_nvme(
        &self,
        state_id: StateId,
        _source: &PhysicalBuffer,
    ) -> Result<(), ExecutorError> {
        // In production: GPU -> CPU (staging) -> NVMe (GDS or file I/O)
        let path = format!("/tmp/aether/state_{}.bin", state_id);
        self.buffers.insert(state_id, PhysicalBuffer::Nvme(path));
        Ok(())
    }

    /// Move data from NVMe to GPU.
    async fn move_nvme_to_gpu(
        &self,
        state_id: StateId,
        source: &PhysicalBuffer,
    ) -> Result<(), ExecutorError> {
        let _path = match source.value() {
            PhysicalBuffer::Nvme(p) => p.clone(),
            _ => return Err(ExecutorError::InvalidAction("Source not on NVMe".to_string())),
        };

        // In production: NVMe -> CPU (staging) -> GPU, or GDS direct
        // Placeholder: create empty GPU buffer
        let state = self.state_manager.get_state(state_id)
            .ok_or(ExecutorError::StateNotFound(state_id))?;

        let gpu_buffer = GpuBuffer::allocate(
            self.device.clone(),
            state.dtype,
            &state.shape,
        ).map_err(|e| ExecutorError::CudaError(e.to_string()))?;

        drop(source);
        self.buffers.insert(state_id, PhysicalBuffer::Gpu(gpu_buffer));

        Ok(())
    }

    /// Move data from CPU to NVMe.
    async fn move_cpu_to_nvme(
        &self,
        state_id: StateId,
        _source: &PhysicalBuffer,
    ) -> Result<(), ExecutorError> {
        let path = format!("/tmp/aether/state_{}.bin", state_id);
        self.buffers.insert(state_id, PhysicalBuffer::Nvme(path));
        Ok(())
    }

    /// Move data from NVMe to CPU.
    async fn move_nvme_to_cpu(
        &self,
        state_id: StateId,
        source: &PhysicalBuffer,
    ) -> Result<(), ExecutorError> {
        let _path = match source.value() {
            PhysicalBuffer::Nvme(p) => p.clone(),
            _ => return Err(ExecutorError::InvalidAction("Source not on NVMe".to_string())),
        };

        let state = self.state_manager.get_state(state_id)
            .ok_or(ExecutorError::StateNotFound(state_id))?;

        let cpu_buffer = CpuBuffer::allocate(state.dtype, &state.shape)
            .map_err(|e| ExecutorError::CudaError(e.to_string()))?;

        drop(source);
        self.buffers.insert(state_id, PhysicalBuffer::Cpu(cpu_buffer));

        Ok(())
    }

    /// Execute PREFETCH action.
    async fn execute_prefetch(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        // Prefetch is like MOVE but initiated speculatively
        debug!("PREFETCH: initiating async fetch for state {}", decision.state_id);
        self.execute_move(decision).await
    }

    /// Execute REPLICATE action.
    async fn execute_replicate(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        // Replicate creates a copy at target without removing source
        debug!("REPLICATE: creating copy of state {}", decision.state_id);
        // In production: copy data to target location
        Ok(())
    }

    /// Execute RECOMPUTE action.
    async fn execute_recompute(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        debug!("RECOMPUTE: regenerating state {}", decision.state_id);

        // In production:
        // 1. Look up the operation that produced this state
        // 2. Ensure input dependencies are available
        // 3. Execute the operation again
        // 4. Store the result

        let mut stats = self.stats.write();
        stats.recomputes_completed += 1;

        Ok(())
    }

    /// Execute COMPRESS action.
    async fn execute_compress(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        debug!("COMPRESS: compressing state {}", decision.state_id);
        // Quantization or other compression
        Ok(())
    }

    /// Execute APPROXIMATE action.
    async fn execute_approximate(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        debug!("APPROXIMATE: approximating state {}", decision.state_id);
        // Low-rank approximation, pruning, etc.
        Ok(())
    }

    /// Execute EVICT action.
    async fn execute_evict(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        let state_id = decision.state_id;
        debug!("EVICT: removing state {}", state_id);

        // Remove physical buffer
        self.buffers.remove(&state_id);

        // Update state manager
        self.state_manager.update_location(state_id, Location::Evicted);

        let mut stats = self.stats.write();
        stats.evictions_completed += 1;

        Ok(())
    }

    /// Execute DELAY action.
    async fn execute_delay(&self, decision: &StateDecision) -> Result<(), ExecutorError> {
        debug!("DELAY: deferring materialization of state {}", decision.state_id);
        // State remains NotMaterialized
        Ok(())
    }

    /// Register a physical buffer for a state.
    pub fn register_buffer(&self, state_id: StateId, buffer: PhysicalBuffer) {
        self.buffers.insert(state_id, buffer);
    }

    /// Get a physical buffer.
    pub fn get_buffer(&self, state_id: StateId) -> Option<PhysicalBuffer> {
        self.buffers.get(&state_id).map(|r| r.value().clone())
    }

    /// Get GPU buffer if available.
    pub fn get_gpu_buffer(&self, state_id: StateId) -> Option<GpuBuffer> {
        self.buffers.get(&state_id).and_then(|r| {
            match r.value() {
                PhysicalBuffer::Gpu(buf) => Some(buf.clone()),
                _ => None,
            }
        })
    }

    /// Get executor statistics.
    pub fn stats(&self) -> ExecutorStats {
        self.stats.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_core::{StateType, DType};

    #[tokio::test]
    async fn test_executor_creation() {
        let device = CudaDevice::new(0).unwrap();
        let device = Arc::new(device);

        let state_manager = Arc::new(StateManager::new(Default::default()));
        let transfer_engine = Arc::new(TransferEngine::new(device.clone()).unwrap());
        let memory_pool = Arc::new(MemoryPool::new(device.clone(), 1024 * 1024 * 1024));
        let stream_pool = Arc::new(StreamPool::new(device.clone(), 4).unwrap());

        let executor = StateExecutor::new(
            state_manager,
            device,
            transfer_engine,
            memory_pool,
            stream_pool,
        );

        assert_eq!(executor.stats().actions_executed, 0);
    }
}
