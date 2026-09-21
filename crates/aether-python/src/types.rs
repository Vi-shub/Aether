//! Python type wrappers for Aether core types.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use aether_core::{StateType, Location, Action, DType, Priority};

/// State type enum.
#[pyclass(name = "StateType")]
#[derive(Clone)]
pub struct PyStateType {
    pub inner: StateType,
}

#[pymethods]
impl PyStateType {
    /// Create a KV cache state type.
    #[staticmethod]
    fn kv_cache(layer: usize) -> Self {
        Self {
            inner: StateType::KvCache { layer },
        }
    }

    /// Create an activation state type.
    #[staticmethod]
    fn activation(layer: usize) -> Self {
        Self {
            inner: StateType::Activation { layer },
        }
    }

    /// Create a weight state type.
    #[staticmethod]
    fn weight(layer: usize) -> Self {
        Self {
            inner: StateType::Weight { layer },
        }
    }

    /// Create an expert weight state type.
    #[staticmethod]
    fn expert_weight(layer: usize, expert: usize) -> Self {
        Self {
            inner: StateType::ExpertWeight { layer, expert },
        }
    }

    /// Create an embedding state type.
    #[staticmethod]
    fn embedding() -> Self {
        Self {
            inner: StateType::Embedding,
        }
    }

    /// Create a speculative state type.
    #[staticmethod]
    fn speculative(branch: usize) -> Self {
        Self {
            inner: StateType::Speculative { branch },
        }
    }

    /// Create an agent context state type.
    #[staticmethod]
    fn agent_context() -> Self {
        Self {
            inner: StateType::AgentContext,
        }
    }

    /// Create a custom state type.
    #[staticmethod]
    fn custom(name: String) -> Self {
        Self {
            inner: StateType::Custom(name),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Memory location enum.
#[pyclass(name = "Location")]
#[derive(Clone)]
pub struct PyLocation {
    pub inner: Location,
}

#[pymethods]
impl PyLocation {
    /// GPU HBM (fastest).
    #[staticmethod]
    fn hbm() -> Self {
        Self { inner: Location::Hbm }
    }

    /// CPU DRAM.
    #[staticmethod]
    fn dram() -> Self {
        Self { inner: Location::Dram }
    }

    /// CXL memory.
    #[staticmethod]
    fn cxl() -> Self {
        Self { inner: Location::Cxl }
    }

    /// NVMe storage.
    #[staticmethod]
    fn nvme() -> Self {
        Self { inner: Location::Nvme }
    }

    /// Remote GPU.
    #[staticmethod]
    fn remote_gpu() -> Self {
        Self { inner: Location::RemoteGpu }
    }

    /// Remote host.
    #[staticmethod]
    fn remote_host() -> Self {
        Self { inner: Location::RemoteHost }
    }

    /// Not yet materialized.
    #[staticmethod]
    fn not_materialized() -> Self {
        Self { inner: Location::NotMaterialized }
    }

    /// Evicted.
    #[staticmethod]
    fn evicted() -> Self {
        Self { inner: Location::Evicted }
    }

    /// Check if the state is materialized.
    fn is_materialized(&self) -> bool {
        self.inner.is_materialized()
    }

    /// Check if on GPU.
    fn is_on_gpu(&self) -> bool {
        self.inner.is_on_gpu()
    }

    /// Check if local (not remote).
    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Action enum for state decisions.
#[pyclass(name = "Action")]
#[derive(Clone)]
pub struct PyAction {
    pub inner: Action,
}

#[pymethods]
impl PyAction {
    /// Keep state in current location.
    #[staticmethod]
    fn keep() -> Self {
        Self { inner: Action::Keep }
    }

    /// Move state to another location.
    #[staticmethod]
    fn move_() -> Self {
        Self { inner: Action::Move }
    }

    /// Prefetch state speculatively.
    #[staticmethod]
    fn prefetch() -> Self {
        Self { inner: Action::Prefetch }
    }

    /// Replicate state to another location.
    #[staticmethod]
    fn replicate() -> Self {
        Self { inner: Action::Replicate }
    }

    /// Recompute state instead of loading.
    #[staticmethod]
    fn recompute() -> Self {
        Self { inner: Action::Recompute }
    }

    /// Compress state.
    #[staticmethod]
    fn compress() -> Self {
        Self { inner: Action::Compress }
    }

    /// Approximate state (quantize, prune).
    #[staticmethod]
    fn approximate() -> Self {
        Self { inner: Action::Approximate }
    }

    /// Evict state from memory.
    #[staticmethod]
    fn evict() -> Self {
        Self { inner: Action::Evict }
    }

    /// Delay materialization.
    #[staticmethod]
    fn delay() -> Self {
        Self { inner: Action::Delay }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Data type enum.
#[pyclass(name = "DType")]
#[derive(Clone)]
pub struct PyDType {
    pub inner: DType,
}

#[pymethods]
impl PyDType {
    #[staticmethod]
    fn float32() -> Self {
        Self { inner: DType::Float32 }
    }

    #[staticmethod]
    fn float16() -> Self {
        Self { inner: DType::Float16 }
    }

    #[staticmethod]
    fn bfloat16() -> Self {
        Self { inner: DType::BFloat16 }
    }

    #[staticmethod]
    fn int8() -> Self {
        Self { inner: DType::Int8 }
    }

    #[staticmethod]
    fn int4() -> Self {
        Self { inner: DType::Int4 }
    }

    /// Get size in bytes per element.
    fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Priority level.
#[pyclass(name = "Priority")]
#[derive(Clone)]
pub struct PyPriority {
    pub inner: Priority,
}

#[pymethods]
impl PyPriority {
    #[staticmethod]
    fn critical() -> Self {
        Self { inner: Priority::Critical }
    }

    #[staticmethod]
    fn high() -> Self {
        Self { inner: Priority::High }
    }

    #[staticmethod]
    fn normal() -> Self {
        Self { inner: Priority::Normal }
    }

    #[staticmethod]
    fn low() -> Self {
        Self { inner: Priority::Low }
    }

    #[staticmethod]
    fn background() -> Self {
        Self { inner: Priority::Background }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}
