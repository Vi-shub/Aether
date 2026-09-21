//! Memory management benchmarks.

use std::sync::Arc;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tracing::info;
use rand::Rng;

use aether_cuda::{CudaDevice, MemoryPool, DeviceMemory};

use crate::scenarios::MemoryScenario;

/// Memory benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    /// Test name.
    pub test_name: String,
    /// Number of allocations.
    pub num_allocations: usize,
    /// Total bytes allocated.
    pub total_bytes: u64,
    /// Time for all allocations (ms).
    pub allocation_time_ms: f64,
    /// Time for all deallocations (ms).
    pub deallocation_time_ms: f64,
    /// Allocations per second.
    pub allocs_per_second: f64,
    /// Peak memory usage.
    pub peak_memory: u64,
    /// Fragmentation ratio (if measured).
    pub fragmentation_ratio: Option<f64>,
}

impl MemoryResult {
    /// Format as table row.
    pub fn to_row(&self) -> Vec<String> {
        vec![
            self.test_name.clone(),
            format!("{}", self.num_allocations),
            format_size(self.total_bytes),
            format!("{:.3}", self.allocation_time_ms),
            format!("{:.3}", self.deallocation_time_ms),
            format!("{:.0}", self.allocs_per_second),
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

/// Memory management benchmark.
pub struct MemoryBenchmark {
    device: Arc<CudaDevice>,
    scenario: MemoryScenario,
}

impl MemoryBenchmark {
    /// Create a new memory benchmark.
    pub fn new(scenario: MemoryScenario) -> Result<Self, anyhow::Error> {
        let device = Arc::new(CudaDevice::new(0)?);

        Ok(Self { device, scenario })
    }

    /// Run all benchmarks.
    pub fn run(&self) -> Vec<MemoryResult> {
        let mut results = Vec::new();

        info!("Starting memory management benchmarks...");

        // Direct allocation benchmark
        results.push(self.benchmark_direct_allocation());

        // Pool allocation benchmark
        results.push(self.benchmark_pool_allocation());

        // Mixed size allocation
        results.push(self.benchmark_mixed_allocation());

        // Allocation/deallocation churn
        results.push(self.benchmark_churn());

        // Fragmentation test
        if self.scenario.test_fragmentation {
            results.push(self.benchmark_fragmentation());
        }

        results
    }

    /// Benchmark direct CUDA allocation.
    fn benchmark_direct_allocation(&self) -> MemoryResult {
        info!("Benchmarking direct allocation...");

        let size = self.scenario.allocation_sizes[0]; // Use first size
        let count = self.scenario.num_allocations.min(100); // Limit for direct alloc
        let mut allocations: Vec<DeviceMemory> = Vec::with_capacity(count);
        let mut total_bytes = 0u64;

        // Allocation phase
        let alloc_start = Instant::now();
        for _ in 0..count {
            if let Ok(mem) = DeviceMemory::allocate(self.device.clone(), size as usize) {
                allocations.push(mem);
                total_bytes += size;
            }
        }
        let alloc_time = alloc_start.elapsed().as_secs_f64() * 1000.0;

        // Deallocation phase
        let dealloc_start = Instant::now();
        drop(allocations);
        let dealloc_time = dealloc_start.elapsed().as_secs_f64() * 1000.0;

        MemoryResult {
            test_name: "Direct Allocation".to_string(),
            num_allocations: count,
            total_bytes,
            allocation_time_ms: alloc_time,
            deallocation_time_ms: dealloc_time,
            allocs_per_second: count as f64 / (alloc_time / 1000.0),
            peak_memory: total_bytes,
            fragmentation_ratio: None,
        }
    }

    /// Benchmark pool allocation.
    fn benchmark_pool_allocation(&self) -> MemoryResult {
        info!("Benchmarking pool allocation...");

        let pool = MemoryPool::new(self.device.clone(), self.scenario.pool_size);
        let size = self.scenario.allocation_sizes[0];
        let count = self.scenario.num_allocations;
        let mut allocations = Vec::with_capacity(count);
        let mut total_bytes = 0u64;

        // Allocation phase
        let alloc_start = Instant::now();
        for _ in 0..count {
            if let Ok(mem) = pool.allocate(size as usize) {
                total_bytes += size;
                allocations.push(mem);
            }
        }
        let alloc_time = alloc_start.elapsed().as_secs_f64() * 1000.0;

        // Deallocation phase
        let dealloc_start = Instant::now();
        drop(allocations);
        let dealloc_time = dealloc_start.elapsed().as_secs_f64() * 1000.0;

        let stats = pool.stats();

        MemoryResult {
            test_name: "Pool Allocation".to_string(),
            num_allocations: count,
            total_bytes,
            allocation_time_ms: alloc_time,
            deallocation_time_ms: dealloc_time,
            allocs_per_second: count as f64 / (alloc_time / 1000.0),
            peak_memory: stats.peak_allocated,
            fragmentation_ratio: None,
        }
    }

    /// Benchmark mixed-size allocation.
    fn benchmark_mixed_allocation(&self) -> MemoryResult {
        info!("Benchmarking mixed-size allocation...");

        let pool = MemoryPool::new(self.device.clone(), self.scenario.pool_size);
        let count = self.scenario.num_allocations;
        let mut allocations = Vec::with_capacity(count);
        let mut total_bytes = 0u64;
        let mut rng = rand::thread_rng();

        // Allocation phase with random sizes
        let alloc_start = Instant::now();
        for _ in 0..count {
            let size_idx = rng.gen_range(0..self.scenario.allocation_sizes.len());
            let size = self.scenario.allocation_sizes[size_idx];

            if let Ok(mem) = pool.allocate(size as usize) {
                total_bytes += size;
                allocations.push(mem);
            }
        }
        let alloc_time = alloc_start.elapsed().as_secs_f64() * 1000.0;

        // Deallocation phase
        let dealloc_start = Instant::now();
        drop(allocations);
        let dealloc_time = dealloc_start.elapsed().as_secs_f64() * 1000.0;

        MemoryResult {
            test_name: "Mixed-Size Allocation".to_string(),
            num_allocations: count,
            total_bytes,
            allocation_time_ms: alloc_time,
            deallocation_time_ms: dealloc_time,
            allocs_per_second: count as f64 / (alloc_time / 1000.0),
            peak_memory: total_bytes,
            fragmentation_ratio: None,
        }
    }

    /// Benchmark allocation/deallocation churn.
    fn benchmark_churn(&self) -> MemoryResult {
        info!("Benchmarking allocation churn...");

        let pool = MemoryPool::new(self.device.clone(), self.scenario.pool_size);
        let size = self.scenario.allocation_sizes[0];
        let count = self.scenario.num_allocations;
        let mut total_bytes = 0u64;
        let mut rng = rand::thread_rng();

        // Keep a working set of allocations
        let mut live: Vec<_> = Vec::new();
        let working_set_size = 100;

        let start = Instant::now();
        for i in 0..count {
            // Randomly allocate or deallocate
            if live.len() < working_set_size || (live.len() < count && rng.gen_bool(0.6)) {
                if let Ok(mem) = pool.allocate(size as usize) {
                    total_bytes += size;
                    live.push(mem);
                }
            } else if !live.is_empty() {
                let idx = rng.gen_range(0..live.len());
                live.swap_remove(idx);
            }
        }

        // Clean up remaining
        drop(live);
        let total_time = start.elapsed().as_secs_f64() * 1000.0;

        let stats = pool.stats();

        MemoryResult {
            test_name: "Allocation Churn".to_string(),
            num_allocations: count,
            total_bytes,
            allocation_time_ms: total_time,
            deallocation_time_ms: 0.0, // Included in total
            allocs_per_second: count as f64 / (total_time / 1000.0),
            peak_memory: stats.peak_allocated,
            fragmentation_ratio: None,
        }
    }

    /// Benchmark fragmentation.
    fn benchmark_fragmentation(&self) -> MemoryResult {
        info!("Benchmarking fragmentation...");

        let pool = MemoryPool::new(self.device.clone(), self.scenario.pool_size);

        // Allocate many small blocks
        let small_size = 1024 * 1024; // 1 MB
        let mut small_allocs: Vec<_> = Vec::new();

        for _ in 0..500 {
            if let Ok(mem) = pool.allocate(small_size) {
                small_allocs.push(mem);
            }
        }

        // Free every other block (create fragmentation)
        let mut remaining: Vec<_> = small_allocs.into_iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, m)| m)
            .collect();

        // Try to allocate larger blocks
        let large_size = 10 * 1024 * 1024; // 10 MB
        let mut large_allocs: Vec<_> = Vec::new();
        let mut large_failures = 0;

        for _ in 0..50 {
            match pool.allocate(large_size) {
                Ok(mem) => large_allocs.push(mem),
                Err(_) => large_failures += 1,
            }
        }

        let stats = pool.stats();
        let fragmentation = if stats.peak_allocated > 0 {
            Some(large_failures as f64 / 50.0)
        } else {
            None
        };

        drop(remaining);
        drop(large_allocs);

        MemoryResult {
            test_name: "Fragmentation Test".to_string(),
            num_allocations: 550,
            total_bytes: stats.bytes_allocated,
            allocation_time_ms: 0.0,
            deallocation_time_ms: 0.0,
            allocs_per_second: 0.0,
            peak_memory: stats.peak_allocated,
            fragmentation_ratio: fragmentation,
        }
    }
}

/// Memory benchmark summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    pub pool_speedup: f64,
    pub avg_allocs_per_second: f64,
    pub peak_memory_used: u64,
}

impl MemorySummary {
    /// Compute summary from results.
    pub fn from_results(results: &[MemoryResult]) -> Self {
        let direct = results.iter()
            .find(|r| r.test_name.contains("Direct"))
            .map(|r| r.allocs_per_second)
            .unwrap_or(1.0);

        let pool = results.iter()
            .find(|r| r.test_name.contains("Pool"))
            .map(|r| r.allocs_per_second)
            .unwrap_or(1.0);

        let peak = results.iter()
            .map(|r| r.peak_memory)
            .max()
            .unwrap_or(0);

        let avg_allocs = results.iter()
            .map(|r| r.allocs_per_second)
            .sum::<f64>() / results.len().max(1) as f64;

        Self {
            pool_speedup: pool / direct,
            avg_allocs_per_second: avg_allocs,
            peak_memory_used: peak,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }
}
