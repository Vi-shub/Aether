//! Hardware configuration for Aether runtime.
//!
//! Defines memory tiers, bandwidths, and compute capabilities
//! that inform the decision engine's cost model.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::Location;

// =============================================================================
// Memory Tier
// =============================================================================

/// Configuration for a single memory tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTier {
    /// Memory location type.
    pub location: Location,
    /// Total capacity in bytes.
    pub capacity_bytes: u64,
    /// Currently allocated bytes.
    pub used_bytes: u64,
    /// Read bandwidth in GB/s.
    pub read_bandwidth_gbps: f64,
    /// Write bandwidth in GB/s.
    pub write_bandwidth_gbps: f64,
    /// Access latency in microseconds.
    pub latency_us: f64,
}

impl MemoryTier {
    /// Create a new memory tier.
    pub fn new(location: Location, capacity_gb: f64, bandwidth_gbps: f64) -> Self {
        Self {
            location,
            capacity_bytes: (capacity_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            used_bytes: 0,
            read_bandwidth_gbps: bandwidth_gbps,
            write_bandwidth_gbps: bandwidth_gbps * 0.9, // Writes typically slower
            latency_us: location.typical_latency_us(),
        }
    }

    /// Capacity in gigabytes.
    #[inline]
    pub fn capacity_gb(&self) -> f64 {
        self.capacity_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Used space in gigabytes.
    #[inline]
    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Available capacity in bytes.
    #[inline]
    pub fn available_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    /// Available capacity in gigabytes.
    #[inline]
    pub fn available_gb(&self) -> f64 {
        self.available_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Memory utilization (0-1).
    #[inline]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Check if size can fit in available space.
    #[inline]
    pub fn can_fit(&self, size_bytes: u64) -> bool {
        size_bytes <= self.available_bytes()
    }

    /// Calculate transfer time in milliseconds.
    pub fn transfer_time_ms(&self, size_bytes: u64, is_read: bool) -> f64 {
        let bw = if is_read {
            self.read_bandwidth_gbps
        } else {
            self.write_bandwidth_gbps
        };

        if bw <= 0.0 {
            return f64::INFINITY;
        }

        // size_bytes / (bw * 1e9) * 1000 = ms
        let transfer_ms = (size_bytes as f64 / (bw * 1e9)) * 1000.0;
        transfer_ms + (self.latency_us / 1000.0)
    }
}

// =============================================================================
// Compute Config
// =============================================================================

/// Compute capability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConfig {
    /// FP16 tensor core TFLOPS.
    pub tflops_fp16: f64,
    /// FP32 TFLOPS.
    pub tflops_fp32: f64,
    /// BF16 TFLOPS.
    pub tflops_bf16: f64,
    /// INT8 TOPS.
    pub tflops_int8: f64,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        // Default to A100-like specs
        Self {
            tflops_fp16: 312.0,
            tflops_fp32: 19.5,
            tflops_bf16: 312.0,
            tflops_int8: 624.0,
        }
    }
}

// =============================================================================
// Hardware Config
// =============================================================================

/// Complete hardware configuration for Aether.
///
/// This captures all hardware characteristics needed for
/// cost modeling and decision making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Device name.
    pub device_name: String,
    /// Number of GPUs.
    pub num_gpus: u32,

    /// GPU HBM capacity in GB.
    pub hbm_capacity_gb: f64,
    /// CPU DRAM capacity in GB.
    pub dram_capacity_gb: f64,
    /// NVMe storage capacity in GB.
    pub nvme_capacity_gb: f64,
    /// CXL memory capacity in GB (0 if not available).
    pub cxl_capacity_gb: f64,

    /// HBM internal bandwidth in GB/s.
    pub hbm_bandwidth_gbps: f64,
    /// PCIe bandwidth in GB/s.
    pub pcie_bandwidth_gbps: f64,
    /// NVMe bandwidth in GB/s.
    pub nvme_bandwidth_gbps: f64,
    /// Network bandwidth in GB/s.
    pub network_bandwidth_gbps: f64,
    /// NVLink bandwidth in GB/s (0 if not available).
    pub nvlink_bandwidth_gbps: f64,

    /// Compute capabilities.
    pub compute: ComputeConfig,

    /// Memory tier configurations.
    #[serde(skip)]
    memory_tiers: HashMap<Location, MemoryTier>,

    /// Bandwidth matrix (source, target) -> GB/s.
    #[serde(skip)]
    bandwidth_matrix: HashMap<(Location, Location), f64>,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareConfig {
    /// Create a new hardware configuration with default values.
    pub fn new() -> Self {
        let mut config = Self {
            device_name: "unknown".to_string(),
            num_gpus: 1,
            hbm_capacity_gb: 80.0,
            dram_capacity_gb: 256.0,
            nvme_capacity_gb: 1000.0,
            cxl_capacity_gb: 0.0,
            hbm_bandwidth_gbps: 2000.0,
            pcie_bandwidth_gbps: 64.0,
            nvme_bandwidth_gbps: 7.0,
            network_bandwidth_gbps: 12.5,
            nvlink_bandwidth_gbps: 0.0,
            compute: ComputeConfig::default(),
            memory_tiers: HashMap::new(),
            bandwidth_matrix: HashMap::new(),
        };
        config.init_tiers();
        config.init_bandwidths();
        config
    }

    /// Create a builder for hardware configuration.
    pub fn builder() -> HardwareConfigBuilder {
        HardwareConfigBuilder::new()
    }

    /// NVIDIA A100 80GB configuration.
    pub fn a100_80gb() -> Self {
        Self::builder()
            .device_name("NVIDIA A100-SXM4-80GB")
            .hbm_capacity_gb(80.0)
            .hbm_bandwidth_gbps(2039.0)
            .pcie_bandwidth_gbps(64.0)
            .compute(ComputeConfig {
                tflops_fp16: 312.0,
                tflops_fp32: 19.5,
                tflops_bf16: 312.0,
                tflops_int8: 624.0,
            })
            .build()
    }

    /// NVIDIA H100 80GB configuration.
    pub fn h100_80gb() -> Self {
        Self::builder()
            .device_name("NVIDIA H100-SXM5-80GB")
            .hbm_capacity_gb(80.0)
            .hbm_bandwidth_gbps(3350.0)
            .pcie_bandwidth_gbps(64.0)
            .nvlink_bandwidth_gbps(900.0)
            .compute(ComputeConfig {
                tflops_fp16: 989.0,
                tflops_fp32: 67.0,
                tflops_bf16: 989.0,
                tflops_int8: 1979.0,
            })
            .build()
    }

    /// NVIDIA RTX 4090 configuration.
    pub fn rtx_4090() -> Self {
        Self::builder()
            .device_name("NVIDIA GeForce RTX 4090")
            .hbm_capacity_gb(24.0)
            .hbm_bandwidth_gbps(1008.0)
            .pcie_bandwidth_gbps(32.0) // PCIe 4.0 x16
            .compute(ComputeConfig {
                tflops_fp16: 165.0,
                tflops_fp32: 82.6,
                tflops_bf16: 165.0,
                tflops_int8: 330.0,
            })
            .build()
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    fn init_tiers(&mut self) {
        self.memory_tiers.clear();

        self.memory_tiers.insert(
            Location::Hbm,
            MemoryTier::new(Location::Hbm, self.hbm_capacity_gb, self.hbm_bandwidth_gbps),
        );

        self.memory_tiers.insert(
            Location::Dram,
            MemoryTier::new(Location::Dram, self.dram_capacity_gb, 100.0), // DDR5
        );

        self.memory_tiers.insert(
            Location::Nvme,
            MemoryTier::new(Location::Nvme, self.nvme_capacity_gb, self.nvme_bandwidth_gbps),
        );

        if self.cxl_capacity_gb > 0.0 {
            self.memory_tiers.insert(
                Location::Cxl,
                MemoryTier::new(Location::Cxl, self.cxl_capacity_gb, 50.0),
            );
        }
    }

    fn init_bandwidths(&mut self) {
        self.bandwidth_matrix.clear();
        let bw = &mut self.bandwidth_matrix;

        // HBM <-> DRAM (via PCIe)
        bw.insert((Location::Hbm, Location::Dram), self.pcie_bandwidth_gbps);
        bw.insert((Location::Dram, Location::Hbm), self.pcie_bandwidth_gbps);

        // DRAM <-> NVMe
        bw.insert((Location::Dram, Location::Nvme), self.nvme_bandwidth_gbps);
        bw.insert((Location::Nvme, Location::Dram), self.nvme_bandwidth_gbps);

        // HBM <-> NVMe (via PCIe or GDS)
        let hbm_nvme_bw = self.pcie_bandwidth_gbps.min(self.nvme_bandwidth_gbps);
        bw.insert((Location::Hbm, Location::Nvme), hbm_nvme_bw);
        bw.insert((Location::Nvme, Location::Hbm), hbm_nvme_bw);

        // CXL paths
        if self.cxl_capacity_gb > 0.0 {
            bw.insert((Location::Dram, Location::Cxl), 50.0);
            bw.insert((Location::Cxl, Location::Dram), 50.0);
            bw.insert((Location::Hbm, Location::Cxl), self.pcie_bandwidth_gbps.min(50.0));
            bw.insert((Location::Cxl, Location::Hbm), self.pcie_bandwidth_gbps.min(50.0));
        }

        // Remote paths
        let remote_bw = if self.nvlink_bandwidth_gbps > 0.0 {
            self.nvlink_bandwidth_gbps
        } else {
            self.network_bandwidth_gbps
        };
        bw.insert((Location::Hbm, Location::RemoteGpu), remote_bw);
        bw.insert((Location::RemoteGpu, Location::Hbm), remote_bw);
        bw.insert((Location::Dram, Location::RemoteHost), self.network_bandwidth_gbps);
        bw.insert((Location::RemoteHost, Location::Dram), self.network_bandwidth_gbps);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get memory tier configuration.
    pub fn get_tier(&self, location: Location) -> Option<&MemoryTier> {
        self.memory_tiers.get(&location)
    }

    /// Get mutable memory tier.
    pub fn get_tier_mut(&mut self, location: Location) -> Option<&mut MemoryTier> {
        self.memory_tiers.get_mut(&location)
    }

    /// Get bandwidth between two locations in GB/s.
    pub fn get_bandwidth(&self, source: Location, target: Location) -> f64 {
        if source == target {
            return f64::INFINITY;
        }
        self.bandwidth_matrix
            .get(&(source, target))
            .copied()
            .unwrap_or(1.0) // Conservative default
    }

    /// Calculate transfer time between locations.
    pub fn transfer_time_ms(&self, source: Location, target: Location, size_bytes: u64) -> f64 {
        let bw = self.get_bandwidth(source, target);
        if bw <= 0.0 || bw == f64::INFINITY {
            return if bw == f64::INFINITY { 0.0 } else { f64::INFINITY };
        }

        // Add latencies
        let src_latency = source.typical_latency_us();
        let tgt_latency = target.typical_latency_us();
        let latency_ms = (src_latency + tgt_latency) / 1000.0;

        (size_bytes as f64 / (bw * 1e9)) * 1000.0 + latency_ms
    }

    /// Get available capacity at a location.
    pub fn available_capacity(&self, location: Location) -> u64 {
        self.memory_tiers
            .get(&location)
            .map(|t| t.available_bytes())
            .unwrap_or(0)
    }

    /// Get total capacity at a location.
    pub fn total_capacity(&self, location: Location) -> u64 {
        self.memory_tiers
            .get(&location)
            .map(|t| t.capacity_bytes)
            .unwrap_or(0)
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("Hardware Configuration: {}", self.device_name),
            format!("  GPUs: {}", self.num_gpus),
            String::new(),
            "Memory Tiers:".to_string(),
        ];

        for (loc, tier) in &self.memory_tiers {
            lines.push(format!(
                "  {:?}: {:.1} GB ({:.0} GB/s read)",
                loc,
                tier.capacity_gb(),
                tier.read_bandwidth_gbps
            ));
        }

        lines.push(String::new());
        lines.push(format!("Compute: {:.0} TFLOPS (FP16)", self.compute.tflops_fp16));

        lines.join("\n")
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for HardwareConfig.
#[derive(Debug, Default)]
pub struct HardwareConfigBuilder {
    device_name: Option<String>,
    num_gpus: Option<u32>,
    hbm_capacity_gb: Option<f64>,
    dram_capacity_gb: Option<f64>,
    nvme_capacity_gb: Option<f64>,
    cxl_capacity_gb: Option<f64>,
    hbm_bandwidth_gbps: Option<f64>,
    pcie_bandwidth_gbps: Option<f64>,
    nvme_bandwidth_gbps: Option<f64>,
    network_bandwidth_gbps: Option<f64>,
    nvlink_bandwidth_gbps: Option<f64>,
    compute: Option<ComputeConfig>,
}

impl HardwareConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn device_name(mut self, name: impl Into<String>) -> Self {
        self.device_name = Some(name.into());
        self
    }

    pub fn num_gpus(mut self, n: u32) -> Self {
        self.num_gpus = Some(n);
        self
    }

    pub fn hbm_capacity_gb(mut self, gb: f64) -> Self {
        self.hbm_capacity_gb = Some(gb);
        self
    }

    pub fn dram_capacity_gb(mut self, gb: f64) -> Self {
        self.dram_capacity_gb = Some(gb);
        self
    }

    pub fn nvme_capacity_gb(mut self, gb: f64) -> Self {
        self.nvme_capacity_gb = Some(gb);
        self
    }

    pub fn cxl_capacity_gb(mut self, gb: f64) -> Self {
        self.cxl_capacity_gb = Some(gb);
        self
    }

    pub fn hbm_bandwidth_gbps(mut self, gbps: f64) -> Self {
        self.hbm_bandwidth_gbps = Some(gbps);
        self
    }

    pub fn pcie_bandwidth_gbps(mut self, gbps: f64) -> Self {
        self.pcie_bandwidth_gbps = Some(gbps);
        self
    }

    pub fn nvme_bandwidth_gbps(mut self, gbps: f64) -> Self {
        self.nvme_bandwidth_gbps = Some(gbps);
        self
    }

    pub fn network_bandwidth_gbps(mut self, gbps: f64) -> Self {
        self.network_bandwidth_gbps = Some(gbps);
        self
    }

    pub fn nvlink_bandwidth_gbps(mut self, gbps: f64) -> Self {
        self.nvlink_bandwidth_gbps = Some(gbps);
        self
    }

    pub fn compute(mut self, config: ComputeConfig) -> Self {
        self.compute = Some(config);
        self
    }

    pub fn build(self) -> HardwareConfig {
        let mut config = HardwareConfig {
            device_name: self.device_name.unwrap_or_else(|| "unknown".to_string()),
            num_gpus: self.num_gpus.unwrap_or(1),
            hbm_capacity_gb: self.hbm_capacity_gb.unwrap_or(80.0),
            dram_capacity_gb: self.dram_capacity_gb.unwrap_or(256.0),
            nvme_capacity_gb: self.nvme_capacity_gb.unwrap_or(1000.0),
            cxl_capacity_gb: self.cxl_capacity_gb.unwrap_or(0.0),
            hbm_bandwidth_gbps: self.hbm_bandwidth_gbps.unwrap_or(2000.0),
            pcie_bandwidth_gbps: self.pcie_bandwidth_gbps.unwrap_or(64.0),
            nvme_bandwidth_gbps: self.nvme_bandwidth_gbps.unwrap_or(7.0),
            network_bandwidth_gbps: self.network_bandwidth_gbps.unwrap_or(12.5),
            nvlink_bandwidth_gbps: self.nvlink_bandwidth_gbps.unwrap_or(0.0),
            compute: self.compute.unwrap_or_default(),
            memory_tiers: HashMap::new(),
            bandwidth_matrix: HashMap::new(),
        };
        config.init_tiers();
        config.init_bandwidths();
        config
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HardwareConfig::new();
        assert!(config.hbm_capacity_gb > 0.0);
        assert!(config.get_tier(Location::Hbm).is_some());
    }

    #[test]
    fn test_a100_config() {
        let config = HardwareConfig::a100_80gb();
        assert_eq!(config.hbm_capacity_gb, 80.0);
        assert!(config.device_name.contains("A100"));
    }

    #[test]
    fn test_transfer_time() {
        let config = HardwareConfig::builder()
            .pcie_bandwidth_gbps(32.0)
            .build();

        // 1GB at 32 GB/s ≈ 31.25ms
        let size = 1024 * 1024 * 1024; // 1GB
        let time = config.transfer_time_ms(Location::Hbm, Location::Dram, size);
        assert!((time - 31.25).abs() < 1.0);
    }

    #[test]
    fn test_memory_tier() {
        let tier = MemoryTier::new(Location::Hbm, 24.0, 1000.0);
        assert_eq!(tier.capacity_gb(), 24.0);
        assert!(tier.can_fit(1024 * 1024 * 1024)); // 1GB fits in 24GB
    }

    #[test]
    fn test_builder() {
        let config = HardwareConfig::builder()
            .device_name("Test GPU")
            .hbm_capacity_gb(16.0)
            .pcie_bandwidth_gbps(16.0)
            .build();

        assert_eq!(config.device_name, "Test GPU");
        assert_eq!(config.hbm_capacity_gb, 16.0);
    }
}
