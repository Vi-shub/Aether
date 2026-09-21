//! Runtime metrics and monitoring.

use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// Runtime metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct RuntimeMetrics {
    // Memory metrics
    pub gpu_memory_allocated: u64,
    pub gpu_memory_available: u64,
    pub gpu_memory_utilization: f64,
    pub cpu_memory_used: u64,
    pub nvme_storage_used: u64,

    // State metrics
    pub total_states: usize,
    pub states_on_gpu: usize,
    pub states_on_cpu: usize,
    pub states_on_nvme: usize,
    pub states_not_materialized: usize,

    // Transfer metrics
    pub pending_transfers: usize,
    pub active_transfers: usize,
    pub bytes_transferred: u64,
    pub transfers_completed: u64,

    // Execution metrics
    pub operations_executed: u64,
    pub total_compute_time_ms: f64,
    pub total_transfer_time_ms: f64,

    // Decision metrics
    pub moves_executed: u64,
    pub recomputes_executed: u64,
    pub evictions_executed: u64,
    pub prefetches_executed: u64,

    // Performance metrics
    pub throughput_tokens_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p99_ms: f64,

    // Efficiency metrics
    pub cache_hit_rate: f64,
    pub memory_efficiency: f64,
    pub compute_utilization: f64,
}

impl RuntimeMetrics {
    /// Calculate memory efficiency (useful work / total memory).
    pub fn calculate_memory_efficiency(&self) -> f64 {
        if self.gpu_memory_allocated == 0 {
            return 0.0;
        }
        // Active states / total allocated
        let active_bytes = (self.states_on_gpu as u64) * 1024; // Placeholder
        active_bytes as f64 / self.gpu_memory_allocated as f64
    }

    /// Format as human-readable string.
    pub fn summary(&self) -> String {
        format!(
            "GPU: {:.1}% ({} / {} GB) | States: {} (GPU: {}, CPU: {}, NVMe: {}) | Transfers: {} pending, {} active | Ops: {}",
            self.gpu_memory_utilization * 100.0,
            self.gpu_memory_allocated / (1024 * 1024 * 1024),
            (self.gpu_memory_allocated + self.gpu_memory_available) / (1024 * 1024 * 1024),
            self.total_states,
            self.states_on_gpu,
            self.states_on_cpu,
            self.states_on_nvme,
            self.pending_transfers,
            self.active_transfers,
            self.operations_executed,
        )
    }
}

/// Latency tracker for P50/P99 calculations.
pub struct LatencyTracker {
    samples: RwLock<Vec<f64>>,
    max_samples: usize,
}

impl LatencyTracker {
    /// Create a new latency tracker.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: RwLock::new(Vec::with_capacity(max_samples)),
            max_samples,
        }
    }

    /// Record a latency sample.
    pub fn record(&self, latency_ms: f64) {
        let mut samples = self.samples.write();
        if samples.len() >= self.max_samples {
            samples.remove(0);
        }
        samples.push(latency_ms);
    }

    /// Get P50 latency.
    pub fn p50(&self) -> f64 {
        self.percentile(0.5)
    }

    /// Get P99 latency.
    pub fn p99(&self) -> f64 {
        self.percentile(0.99)
    }

    /// Get percentile.
    fn percentile(&self, p: f64) -> f64 {
        let samples = self.samples.read();
        if samples.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f64) * p) as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Clear samples.
    pub fn clear(&self) {
        self.samples.write().clear();
    }
}

/// Throughput tracker.
pub struct ThroughputTracker {
    window_start: RwLock<Instant>,
    tokens_in_window: RwLock<u64>,
    window_duration: Duration,
    last_throughput: RwLock<f64>,
}

impl ThroughputTracker {
    /// Create a new throughput tracker.
    pub fn new(window_duration: Duration) -> Self {
        Self {
            window_start: RwLock::new(Instant::now()),
            tokens_in_window: RwLock::new(0),
            window_duration,
            last_throughput: RwLock::new(0.0),
        }
    }

    /// Record tokens processed.
    pub fn record(&self, tokens: u64) {
        let mut count = self.tokens_in_window.write();
        *count += tokens;

        // Check if window expired
        let start = *self.window_start.read();
        if start.elapsed() >= self.window_duration {
            let duration_secs = start.elapsed().as_secs_f64();
            *self.last_throughput.write() = *count as f64 / duration_secs;

            // Reset window
            *self.window_start.write() = Instant::now();
            *count = 0;
        }
    }

    /// Get current throughput.
    pub fn throughput(&self) -> f64 {
        *self.last_throughput.read()
    }
}

/// Metrics collector aggregates all runtime metrics.
pub struct MetricsCollector {
    latency_tracker: LatencyTracker,
    throughput_tracker: ThroughputTracker,
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            latency_tracker: LatencyTracker::new(10000),
            throughput_tracker: ThroughputTracker::new(Duration::from_secs(1)),
            start_time: Instant::now(),
        }
    }

    /// Record a completed operation.
    pub fn record_operation(&self, latency_ms: f64, tokens: u64) {
        self.latency_tracker.record(latency_ms);
        self.throughput_tracker.record(tokens);
    }

    /// Collect current metrics from runtime.
    pub fn collect(&self, runtime: &super::UnifiedRuntime) -> RuntimeMetrics {
        let pool_stats = runtime.memory_pool.stats();
        let transfer_stats = runtime.transfer_scheduler.stats();
        let exec_stats = runtime.state_executor.stats();
        let kernel_stats = runtime.kernel_executor.stats();

        // Get state counts by location
        let states = runtime.state_manager.all_states();
        let states_on_gpu = states.iter()
            .filter(|s| s.location == aether_core::Location::Hbm)
            .count();
        let states_on_cpu = states.iter()
            .filter(|s| s.location == aether_core::Location::Dram)
            .count();
        let states_on_nvme = states.iter()
            .filter(|s| s.location == aether_core::Location::Nvme)
            .count();
        let states_not_materialized = states.iter()
            .filter(|s| s.location == aether_core::Location::NotMaterialized)
            .count();

        let gpu_total = pool_stats.total_capacity;
        let gpu_allocated = pool_stats.bytes_allocated;

        RuntimeMetrics {
            gpu_memory_allocated: gpu_allocated,
            gpu_memory_available: gpu_total.saturating_sub(gpu_allocated),
            gpu_memory_utilization: if gpu_total > 0 {
                gpu_allocated as f64 / gpu_total as f64
            } else {
                0.0
            },
            cpu_memory_used: 0, // Would need OS query
            nvme_storage_used: 0,

            total_states: states.len(),
            states_on_gpu,
            states_on_cpu,
            states_on_nvme,
            states_not_materialized,

            pending_transfers: runtime.transfer_scheduler.pending_count(),
            active_transfers: runtime.transfer_scheduler.active_count(),
            bytes_transferred: transfer_stats.bytes_transferred,
            transfers_completed: transfer_stats.tasks_completed,

            operations_executed: kernel_stats.operations_executed,
            total_compute_time_ms: kernel_stats.total_compute_time_ms,
            total_transfer_time_ms: transfer_stats.total_transfer_time_ms,

            moves_executed: exec_stats.moves_completed,
            recomputes_executed: exec_stats.recomputes_completed,
            evictions_executed: exec_stats.evictions_completed,
            prefetches_executed: 0,

            throughput_tokens_per_sec: self.throughput_tracker.throughput(),
            latency_p50_ms: self.latency_tracker.p50(),
            latency_p99_ms: self.latency_tracker.p99(),

            cache_hit_rate: if kernel_stats.cache_hits + kernel_stats.cache_misses > 0 {
                kernel_stats.cache_hits as f64 /
                    (kernel_stats.cache_hits + kernel_stats.cache_misses) as f64
            } else {
                1.0
            },
            memory_efficiency: 0.0, // Would need more detailed tracking
            compute_utilization: 0.0, // Would need GPU profiling
        }
    }

    /// Get uptime.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_tracker() {
        let tracker = LatencyTracker::new(100);

        for i in 0..100 {
            tracker.record(i as f64);
        }

        // P50 should be around 50
        let p50 = tracker.p50();
        assert!(p50 >= 45.0 && p50 <= 55.0);
    }

    #[test]
    fn test_metrics_summary() {
        let metrics = RuntimeMetrics {
            gpu_memory_allocated: 8 * 1024 * 1024 * 1024,
            gpu_memory_available: 32 * 1024 * 1024 * 1024,
            gpu_memory_utilization: 0.2,
            total_states: 100,
            states_on_gpu: 50,
            states_on_cpu: 30,
            states_on_nvme: 20,
            pending_transfers: 5,
            active_transfers: 2,
            operations_executed: 1000,
            ..Default::default()
        };

        let summary = metrics.summary();
        assert!(summary.contains("GPU:"));
        assert!(summary.contains("States:"));
    }
}
