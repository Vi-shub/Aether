//! Python wrappers for hardware configuration.

use pyo3::prelude::*;
use aether_core::{HardwareConfig, MemoryTier};

/// Memory tier configuration.
#[pyclass(name = "MemoryTier")]
#[derive(Clone)]
pub struct PyMemoryTier {
    pub inner: MemoryTier,
}

#[pymethods]
impl PyMemoryTier {
    /// Create a new memory tier.
    #[new]
    fn new(
        capacity_bytes: u64,
        bandwidth_gbps: f64,
        latency_us: f64,
    ) -> Self {
        Self {
            inner: MemoryTier {
                capacity_bytes,
                bandwidth_gbps,
                latency_us,
            },
        }
    }

    /// Create HBM tier (H100-like).
    #[staticmethod]
    fn hbm_h100() -> Self {
        Self {
            inner: MemoryTier {
                capacity_bytes: 80 * 1024 * 1024 * 1024,
                bandwidth_gbps: 3350.0,
                latency_us: 0.1,
            },
        }
    }

    /// Create HBM tier (A100-like).
    #[staticmethod]
    fn hbm_a100() -> Self {
        Self {
            inner: MemoryTier {
                capacity_bytes: 80 * 1024 * 1024 * 1024,
                bandwidth_gbps: 2039.0,
                latency_us: 0.1,
            },
        }
    }

    /// Create DRAM tier.
    #[staticmethod]
    fn dram(capacity_gb: u64) -> Self {
        Self {
            inner: MemoryTier {
                capacity_bytes: capacity_gb * 1024 * 1024 * 1024,
                bandwidth_gbps: 200.0,
                latency_us: 0.1,
            },
        }
    }

    /// Create NVMe tier.
    #[staticmethod]
    fn nvme(capacity_gb: u64) -> Self {
        Self {
            inner: MemoryTier {
                capacity_bytes: capacity_gb * 1024 * 1024 * 1024,
                bandwidth_gbps: 7.0,
                latency_us: 10.0,
            },
        }
    }

    #[getter]
    fn capacity_bytes(&self) -> u64 {
        self.inner.capacity_bytes
    }

    #[getter]
    fn capacity_gb(&self) -> f64 {
        self.inner.capacity_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    #[getter]
    fn bandwidth_gbps(&self) -> f64 {
        self.inner.bandwidth_gbps
    }

    #[getter]
    fn latency_us(&self) -> f64 {
        self.inner.latency_us
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryTier(capacity={:.1}GB, bandwidth={:.1}GB/s, latency={:.1}μs)",
            self.capacity_gb(),
            self.inner.bandwidth_gbps,
            self.inner.latency_us
        )
    }
}

/// Hardware configuration.
#[pyclass(name = "HardwareConfig")]
#[derive(Clone)]
pub struct PyHardwareConfig {
    pub inner: HardwareConfig,
}

#[pymethods]
impl PyHardwareConfig {
    /// Create a new hardware configuration.
    #[new]
    #[pyo3(signature = (
        hbm = None,
        dram = None,
        nvme = None,
        compute_tflops = 1000.0,
        pcie_bandwidth_gbps = 32.0
    ))]
    fn new(
        hbm: Option<PyMemoryTier>,
        dram: Option<PyMemoryTier>,
        nvme: Option<PyMemoryTier>,
        compute_tflops: f64,
        pcie_bandwidth_gbps: f64,
    ) -> Self {
        let mut config = HardwareConfig::default();

        if let Some(h) = hbm {
            config.hbm = h.inner;
        }
        if let Some(d) = dram {
            config.dram = d.inner;
        }
        if let Some(n) = nvme {
            config.nvme = Some(n.inner);
        }

        config.compute.fp16_tflops = compute_tflops;
        config.pcie_bandwidth_gbps = pcie_bandwidth_gbps;

        Self { inner: config }
    }

    /// Create H100 configuration.
    #[staticmethod]
    fn h100() -> Self {
        Self {
            inner: HardwareConfig::h100(),
        }
    }

    /// Create A100 configuration.
    #[staticmethod]
    fn a100() -> Self {
        Self {
            inner: HardwareConfig::a100(),
        }
    }

    /// Create RTX 4090 configuration.
    #[staticmethod]
    fn rtx_4090() -> Self {
        Self {
            inner: HardwareConfig::rtx_4090(),
        }
    }

    /// Create a custom configuration with given GPU memory.
    #[staticmethod]
    fn custom(gpu_memory_gb: u64, cpu_memory_gb: u64) -> Self {
        let mut config = HardwareConfig::default();
        config.hbm.capacity_bytes = gpu_memory_gb * 1024 * 1024 * 1024;
        config.dram.capacity_bytes = cpu_memory_gb * 1024 * 1024 * 1024;
        Self { inner: config }
    }

    #[getter]
    fn hbm(&self) -> PyMemoryTier {
        PyMemoryTier { inner: self.inner.hbm.clone() }
    }

    #[getter]
    fn dram(&self) -> PyMemoryTier {
        PyMemoryTier { inner: self.inner.dram.clone() }
    }

    #[getter]
    fn nvme(&self) -> Option<PyMemoryTier> {
        self.inner.nvme.clone().map(|n| PyMemoryTier { inner: n })
    }

    #[getter]
    fn compute_tflops(&self) -> f64 {
        self.inner.compute.fp16_tflops
    }

    #[getter]
    fn pcie_bandwidth_gbps(&self) -> f64 {
        self.inner.pcie_bandwidth_gbps
    }

    /// Calculate transfer time between locations.
    fn transfer_time_ms(&self, from_location: &str, to_location: &str, size_bytes: u64) -> f64 {
        let bandwidth = match (from_location, to_location) {
            ("hbm", "dram") | ("dram", "hbm") => self.inner.pcie_bandwidth_gbps,
            ("dram", "nvme") | ("nvme", "dram") => {
                self.inner.nvme.as_ref().map(|n| n.bandwidth_gbps).unwrap_or(7.0)
            }
            ("hbm", "hbm") => self.inner.hbm.bandwidth_gbps,
            _ => self.inner.pcie_bandwidth_gbps,
        };

        let size_gb = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        size_gb / bandwidth * 1000.0
    }

    fn __repr__(&self) -> String {
        format!(
            "HardwareConfig(HBM={:.1}GB, DRAM={:.1}GB, compute={:.0}TFLOPS)",
            self.inner.hbm.capacity_bytes as f64 / 1e9,
            self.inner.dram.capacity_bytes as f64 / 1e9,
            self.inner.compute.fp16_tflops
        )
    }
}
