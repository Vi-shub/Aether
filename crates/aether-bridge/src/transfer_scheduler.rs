//! Transfer scheduler - manages and prioritizes data transfers.

use std::sync::Arc;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use aether_core::{Location, StateId};
use aether_cuda::{CudaDevice, CudaStream, TransferEngine, StreamPool, TransferHandle};

/// Transfer priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    /// Critical path - blocking execution.
    Critical = 4,
    /// High priority - needed soon.
    High = 3,
    /// Normal priority.
    Normal = 2,
    /// Background - opportunistic.
    Background = 1,
    /// Prefetch - speculative.
    Prefetch = 0,
}

/// A scheduled transfer task.
#[derive(Debug, Clone)]
pub struct TransferTask {
    /// Unique task ID.
    pub id: u64,
    /// State being transferred.
    pub state_id: StateId,
    /// Source location.
    pub source: Location,
    /// Destination location.
    pub destination: Location,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Priority.
    pub priority: TransferPriority,
    /// Deadline (optional, in milliseconds from now).
    pub deadline_ms: Option<f64>,
}

impl TransferTask {
    /// Create a new transfer task.
    pub fn new(
        state_id: StateId,
        source: Location,
        destination: Location,
        size_bytes: u64,
    ) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        Self {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            state_id,
            source,
            destination,
            size_bytes,
            priority: TransferPriority::Normal,
            deadline_ms: None,
        }
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: TransferPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set deadline.
    pub fn with_deadline(mut self, deadline_ms: f64) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }

    /// Check if this is a critical transfer.
    pub fn is_critical(&self) -> bool {
        self.priority == TransferPriority::Critical
    }

    /// Estimate transfer time based on bandwidth.
    pub fn estimate_time(&self, bandwidth_gbps: f64) -> f64 {
        let bandwidth_bps = bandwidth_gbps * 1e9;
        self.size_bytes as f64 / bandwidth_bps * 1000.0 // ms
    }
}

// Priority queue ordering for TransferTask
impl Eq for TransferTask {}

impl PartialEq for TransferTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for TransferTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier deadline
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                match (self.deadline_ms, other.deadline_ms) {
                    (Some(a), Some(b)) => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
                    (Some(_), None) => Ordering::Greater,
                    (None, Some(_)) => Ordering::Less,
                    (None, None) => Ordering::Equal,
                }
            }
            ord => ord,
        }
    }
}

impl PartialOrd for TransferTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Active transfer tracking.
#[derive(Debug)]
struct ActiveTransfer {
    task: TransferTask,
    handle: TransferHandle,
    started_at: std::time::Instant,
}

/// Transfer scheduler manages concurrent data transfers.
pub struct TransferScheduler {
    /// CUDA device.
    device: Arc<CudaDevice>,
    /// Transfer engine.
    transfer_engine: Arc<TransferEngine>,
    /// Stream pool.
    stream_pool: Arc<StreamPool>,
    /// Pending tasks (priority queue).
    pending: Mutex<BinaryHeap<TransferTask>>,
    /// Active transfers.
    active: Mutex<Vec<ActiveTransfer>>,
    /// Maximum concurrent transfers.
    max_concurrent: usize,
    /// Statistics.
    stats: Mutex<SchedulerStats>,
}

/// Scheduler statistics.
#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_cancelled: u64,
    pub bytes_transferred: u64,
    pub total_wait_time_ms: f64,
    pub total_transfer_time_ms: f64,
}

impl TransferScheduler {
    /// Create a new transfer scheduler.
    pub fn new(
        device: Arc<CudaDevice>,
        transfer_engine: Arc<TransferEngine>,
        stream_pool: Arc<StreamPool>,
        max_concurrent: usize,
    ) -> Self {
        Self {
            device,
            transfer_engine,
            stream_pool,
            pending: Mutex::new(BinaryHeap::new()),
            active: Mutex::new(Vec::new()),
            max_concurrent,
            stats: Mutex::new(SchedulerStats::default()),
        }
    }

    /// Submit a transfer task.
    pub fn submit(&self, task: TransferTask) -> u64 {
        let id = task.id;
        debug!(
            "Submitted transfer: id={}, state={}, {:?} -> {:?}, {} bytes, priority={:?}",
            id, task.state_id, task.source, task.destination, task.size_bytes, task.priority
        );

        self.pending.lock().push(task);
        self.stats.lock().tasks_submitted += 1;

        // Try to start transfers
        self.try_start_transfers();

        id
    }

    /// Submit multiple tasks.
    pub fn submit_batch(&self, tasks: Vec<TransferTask>) -> Vec<u64> {
        let ids: Vec<u64> = tasks.iter().map(|t| t.id).collect();

        {
            let mut pending = self.pending.lock();
            for task in tasks {
                pending.push(task);
            }
        }

        self.stats.lock().tasks_submitted += ids.len() as u64;
        self.try_start_transfers();

        ids
    }

    /// Try to start pending transfers.
    fn try_start_transfers(&self) {
        let mut pending = self.pending.lock();
        let mut active = self.active.lock();

        while active.len() < self.max_concurrent {
            if let Some(task) = pending.pop() {
                match self.start_transfer(&task) {
                    Ok(handle) => {
                        active.push(ActiveTransfer {
                            task,
                            handle,
                            started_at: std::time::Instant::now(),
                        });
                    }
                    Err(e) => {
                        warn!("Failed to start transfer {}: {}", task.id, e);
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Start a single transfer.
    fn start_transfer(&self, task: &TransferTask) -> Result<TransferHandle, String> {
        debug!(
            "Starting transfer: {:?} -> {:?}, {} bytes",
            task.source, task.destination, task.size_bytes
        );

        // Get a stream
        let stream = self.stream_pool.acquire()
            .map_err(|e| e.to_string())?;

        // The actual transfer would be initiated here
        // For now, create a mock handle
        let handle = TransferHandle::mock(task.id, stream);

        Ok(handle)
    }

    /// Poll for completed transfers.
    pub fn poll_completions(&self) -> Vec<TransferTask> {
        let mut completed = Vec::new();
        let mut active = self.active.lock();

        active.retain(|transfer| {
            if transfer.handle.is_complete() {
                let elapsed = transfer.started_at.elapsed().as_secs_f64() * 1000.0;

                debug!(
                    "Transfer completed: id={}, took {:.2}ms",
                    transfer.task.id, elapsed
                );

                let mut stats = self.stats.lock();
                stats.tasks_completed += 1;
                stats.bytes_transferred += transfer.task.size_bytes;
                stats.total_transfer_time_ms += elapsed;

                completed.push(transfer.task.clone());
                false
            } else {
                true
            }
        });

        // Try to start more transfers
        if !completed.is_empty() {
            drop(active);
            self.try_start_transfers();
        }

        completed
    }

    /// Wait for a specific transfer to complete.
    pub async fn wait_for(&self, task_id: u64) -> Option<TransferTask> {
        loop {
            let completed = self.poll_completions();
            for task in &completed {
                if task.id == task_id {
                    return Some(task.clone());
                }
            }

            // Check if task is still pending or active
            let in_pending = self.pending.lock().iter().any(|t| t.id == task_id);
            let in_active = self.active.lock().iter().any(|t| t.task.id == task_id);

            if !in_pending && !in_active {
                // Task not found - already completed or never submitted
                return None;
            }

            // Small sleep before polling again
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }
    }

    /// Wait for all pending transfers.
    pub async fn wait_all(&self) {
        loop {
            self.poll_completions();

            let pending_count = self.pending.lock().len();
            let active_count = self.active.lock().len();

            if pending_count == 0 && active_count == 0 {
                break;
            }

            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }
    }

    /// Cancel a pending transfer.
    pub fn cancel(&self, task_id: u64) -> bool {
        let mut pending = self.pending.lock();

        // Find and remove from pending
        let original_len = pending.len();
        let new_pending: BinaryHeap<_> = pending
            .drain()
            .filter(|t| t.id != task_id)
            .collect();

        let removed = new_pending.len() < original_len;
        *pending = new_pending;

        if removed {
            self.stats.lock().tasks_cancelled += 1;
        }

        removed
    }

    /// Get number of pending tasks.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Get number of active transfers.
    pub fn active_count(&self) -> usize {
        self.active.lock().len()
    }

    /// Get scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        self.stats.lock().clone()
    }

    /// Calculate optimal transfer order for a set of tasks.
    pub fn optimize_order(&self, tasks: &mut [TransferTask]) {
        // Sort by:
        // 1. Critical path tasks first
        // 2. Tasks with deadlines
        // 3. Smaller transfers (to reduce head-of-line blocking)
        tasks.sort_by(|a, b| {
            match b.priority.cmp(&a.priority) {
                Ordering::Equal => {
                    match (a.deadline_ms, b.deadline_ms) {
                        (Some(da), Some(db)) => da.partial_cmp(&db).unwrap_or(Ordering::Equal),
                        (Some(_), None) => Ordering::Less,
                        (None, Some(_)) => Ordering::Greater,
                        (None, None) => a.size_bytes.cmp(&b.size_bytes),
                    }
                }
                ord => ord,
            }
        });
    }
}

/// Transfer handle for tracking async transfers.
impl TransferHandle {
    /// Create a mock handle for testing.
    pub fn mock(id: u64, stream: Arc<CudaStream>) -> Self {
        Self {
            request_id: id,
            stream,
        }
    }

    /// Check if transfer is complete.
    pub fn is_complete(&self) -> bool {
        // In production, this would query CUDA stream/event
        true // Mock: always complete
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_task_priority() {
        let critical = TransferTask::new(0, Location::Dram, Location::Hbm, 1000)
            .with_priority(TransferPriority::Critical);

        let normal = TransferTask::new(1, Location::Dram, Location::Hbm, 1000)
            .with_priority(TransferPriority::Normal);

        assert!(critical > normal);
    }

    #[test]
    fn test_transfer_task_deadline() {
        let urgent = TransferTask::new(0, Location::Dram, Location::Hbm, 1000)
            .with_deadline(10.0);

        let relaxed = TransferTask::new(1, Location::Dram, Location::Hbm, 1000)
            .with_deadline(100.0);

        // Same priority, earlier deadline wins
        assert!(urgent > relaxed);
    }
}
