//! Transfer performance benchmarks.

use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use aether_cuda::{CudaDevice, TransferEngine, StreamPool, DeviceMemory, PinnedMemory};

use crate::scenarios::TransferScenario;

/// Transfer benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferResult {
    /// Transfer direction.
    pub direction: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Time in milliseconds.
    pub time_ms: f64,
    /// Bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Standard deviation of time.
    pub time_stddev_ms: f64,
}

impl TransferResult {
    /// Format as table row.
    pub fn to_row(&self) -> Vec<String> {
        vec![
            self.direction.clone(),
            format_size(self.size_bytes),
            format!("{:.3}", self.time_ms),
            format!("{:.2}", self.bandwidth_gbps),
            format!("±{:.3}", self.time_stddev_ms),
        ]
    }
}

/// Format bytes as human-readable size.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Transfer benchmark runner.
pub struct TransferBenchmark {
    device: Arc<CudaDevice>,
    transfer_engine: Arc<TransferEngine>,
    stream_pool: Arc<StreamPool>,
    scenario: TransferScenario,
}

impl TransferBenchmark {
    /// Create a new transfer benchmark.
    pub fn new(scenario: TransferScenario) -> Result<Self, anyhow::Error> {
        let device = Arc::new(CudaDevice::new(0)?);
        let transfer_engine = Arc::new(TransferEngine::new(device.clone())?);
        let stream_pool = Arc::new(StreamPool::new(device.clone(), 4)?);

        Ok(Self {
            device,
            transfer_engine,
            stream_pool,
            scenario,
        })
    }

    /// Run all benchmarks.
    pub fn run(&self) -> Vec<TransferResult> {
        let mut results = Vec::new();

        info!("Starting transfer benchmarks...");

        for &size in &self.scenario.sizes {
            info!("Benchmarking size: {}", format_size(size));

            // Warmup
            for _ in 0..self.scenario.warmup {
                self.run_single_transfer("warmup", size);
            }

            // H2D
            if self.scenario.test_h2d {
                let times = self.benchmark_h2d(size);
                results.push(self.compute_result("H2D", size, &times));
            }

            // D2H
            if self.scenario.test_d2h {
                let times = self.benchmark_d2h(size);
                results.push(self.compute_result("D2H", size, &times));
            }

            // D2D
            if self.scenario.test_d2d {
                let times = self.benchmark_d2d(size);
                results.push(self.compute_result("D2D", size, &times));
            }
        }

        // Concurrent transfers
        if self.scenario.test_concurrent {
            info!("Benchmarking concurrent transfers...");
            for &size in &self.scenario.sizes {
                let times = self.benchmark_concurrent(size, 4);
                results.push(self.compute_result("H2D (4x concurrent)", size, &times));
            }
        }

        results
    }

    /// Benchmark host-to-device transfers.
    fn benchmark_h2d(&self, size: u64) -> Vec<f64> {
        let mut times = Vec::with_capacity(self.scenario.iterations);

        for _ in 0..self.scenario.iterations {
            let time = self.run_h2d_transfer(size);
            times.push(time.as_secs_f64() * 1000.0);
        }

        times
    }

    /// Benchmark device-to-host transfers.
    fn benchmark_d2h(&self, size: u64) -> Vec<f64> {
        let mut times = Vec::with_capacity(self.scenario.iterations);

        for _ in 0..self.scenario.iterations {
            let time = self.run_d2h_transfer(size);
            times.push(time.as_secs_f64() * 1000.0);
        }

        times
    }

    /// Benchmark device-to-device transfers.
    fn benchmark_d2d(&self, size: u64) -> Vec<f64> {
        let mut times = Vec::with_capacity(self.scenario.iterations);

        for _ in 0..self.scenario.iterations {
            let time = self.run_d2d_transfer(size);
            times.push(time.as_secs_f64() * 1000.0);
        }

        times
    }

    /// Benchmark concurrent transfers.
    fn benchmark_concurrent(&self, size: u64, num_concurrent: usize) -> Vec<f64> {
        let mut times = Vec::with_capacity(self.scenario.iterations);

        for _ in 0..self.scenario.iterations {
            let time = self.run_concurrent_transfers(size, num_concurrent);
            times.push(time.as_secs_f64() * 1000.0);
        }

        times
    }

    /// Run a single H2D transfer.
    fn run_h2d_transfer(&self, size: u64) -> Duration {
        // Allocate pinned host memory
        let host_data: Vec<u8> = vec![0u8; size as usize];

        // Allocate device memory
        let mut device_mem = DeviceMemory::allocate(self.device.clone(), size as usize)
            .expect("Failed to allocate device memory");

        // Synchronize before timing
        self.device.synchronize().ok();

        let start = Instant::now();

        // Perform transfer
        self.transfer_engine
            .copy_h2d_async(&host_data, &mut device_mem, None)
            .expect("H2D transfer failed");

        // Synchronize after
        self.device.synchronize().ok();

        start.elapsed()
    }

    /// Run a single D2H transfer.
    fn run_d2h_transfer(&self, size: u64) -> Duration {
        // Allocate and initialize device memory
        let device_mem = DeviceMemory::allocate(self.device.clone(), size as usize)
            .expect("Failed to allocate device memory");

        // Allocate host memory
        let mut host_data: Vec<u8> = vec![0u8; size as usize];

        self.device.synchronize().ok();

        let start = Instant::now();

        // Perform transfer
        self.transfer_engine
            .copy_d2h_async(&device_mem, &mut host_data, None)
            .expect("D2H transfer failed");

        self.device.synchronize().ok();

        start.elapsed()
    }

    /// Run a single D2D transfer.
    fn run_d2d_transfer(&self, size: u64) -> Duration {
        // Allocate two device buffers
        let src = DeviceMemory::allocate(self.device.clone(), size as usize)
            .expect("Failed to allocate source");
        let mut dst = DeviceMemory::allocate(self.device.clone(), size as usize)
            .expect("Failed to allocate destination");

        self.device.synchronize().ok();

        let start = Instant::now();

        // Perform transfer
        self.transfer_engine
            .copy_d2d_async(&src, &mut dst, None)
            .expect("D2D transfer failed");

        self.device.synchronize().ok();

        start.elapsed()
    }

    /// Run concurrent transfers.
    fn run_concurrent_transfers(&self, size: u64, num_concurrent: usize) -> Duration {
        // Allocate buffers for concurrent transfers
        let host_buffers: Vec<Vec<u8>> = (0..num_concurrent)
            .map(|_| vec![0u8; size as usize])
            .collect();

        let mut device_buffers: Vec<DeviceMemory> = (0..num_concurrent)
            .map(|_| DeviceMemory::allocate(self.device.clone(), size as usize)
                .expect("Failed to allocate"))
            .collect();

        self.device.synchronize().ok();

        let start = Instant::now();

        // Launch all transfers
        for (host, device) in host_buffers.iter().zip(device_buffers.iter_mut()) {
            self.transfer_engine
                .copy_h2d_async(host, device, None)
                .expect("Concurrent transfer failed");
        }

        self.device.synchronize().ok();

        start.elapsed()
    }

    /// Run a single transfer (for warmup).
    fn run_single_transfer(&self, _name: &str, size: u64) {
        let _ = self.run_h2d_transfer(size);
    }

    /// Compute result statistics.
    fn compute_result(&self, direction: &str, size: u64, times: &[f64]) -> TransferResult {
        let mean = times.iter().sum::<f64>() / times.len() as f64;

        let variance = times.iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>() / times.len() as f64;
        let stddev = variance.sqrt();

        let bandwidth = (size as f64 / (1024.0 * 1024.0 * 1024.0)) / (mean / 1000.0);

        TransferResult {
            direction: direction.to_string(),
            size_bytes: size,
            time_ms: mean,
            bandwidth_gbps: bandwidth,
            time_stddev_ms: stddev,
        }
    }
}

/// Summary statistics for transfer benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferSummary {
    pub peak_h2d_bandwidth: f64,
    pub peak_d2h_bandwidth: f64,
    pub peak_d2d_bandwidth: f64,
    pub avg_h2d_bandwidth: f64,
    pub avg_d2h_bandwidth: f64,
    pub avg_d2d_bandwidth: f64,
}

impl TransferSummary {
    /// Compute summary from results.
    pub fn from_results(results: &[TransferResult]) -> Self {
        let h2d_results: Vec<_> = results.iter()
            .filter(|r| r.direction == "H2D")
            .collect();
        let d2h_results: Vec<_> = results.iter()
            .filter(|r| r.direction == "D2H")
            .collect();
        let d2d_results: Vec<_> = results.iter()
            .filter(|r| r.direction == "D2D")
            .collect();

        Self {
            peak_h2d_bandwidth: h2d_results.iter()
                .map(|r| r.bandwidth_gbps)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0),
            peak_d2h_bandwidth: d2h_results.iter()
                .map(|r| r.bandwidth_gbps)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0),
            peak_d2d_bandwidth: d2d_results.iter()
                .map(|r| r.bandwidth_gbps)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0),
            avg_h2d_bandwidth: h2d_results.iter()
                .map(|r| r.bandwidth_gbps)
                .sum::<f64>() / h2d_results.len().max(1) as f64,
            avg_d2h_bandwidth: d2h_results.iter()
                .map(|r| r.bandwidth_gbps)
                .sum::<f64>() / d2h_results.len().max(1) as f64,
            avg_d2d_bandwidth: d2d_results.iter()
                .map(|r| r.bandwidth_gbps)
                .sum::<f64>() / d2d_results.len().max(1) as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_compute_result() {
        let times = vec![10.0, 11.0, 9.0, 10.5, 9.5];
        let size = 1024 * 1024 * 1024; // 1 GB

        let benchmark = TransferBenchmark::new(TransferScenario::default());
        if let Ok(b) = benchmark {
            let result = b.compute_result("H2D", size, &times);
            assert!(result.time_ms > 0.0);
            assert!(result.bandwidth_gbps > 0.0);
        }
    }
}
