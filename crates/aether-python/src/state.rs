//! Python wrapper for ExecutionState.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use aether_core::{ExecutionState, StateType, Location, DType, Priority};

use crate::types::{PyStateType, PyLocation, PyDType, PyPriority};

/// Execution state - the fundamental unit in Aether.
///
/// Represents any piece of state in AI inference: KV cache, activations,
/// weights, expert parameters, etc.
#[pyclass(name = "State")]
#[derive(Clone)]
pub struct PyState {
    pub inner: ExecutionState,
}

#[pymethods]
impl PyState {
    /// Create a new state.
    #[new]
    #[pyo3(signature = (
        id,
        state_type,
        name,
        size_bytes,
        shape,
        location = None,
        dtype = None,
        priority = None,
        recomputable = false,
        pinned = false
    ))]
    fn new(
        id: u64,
        state_type: PyStateType,
        name: String,
        size_bytes: u64,
        shape: Vec<usize>,
        location: Option<PyLocation>,
        dtype: Option<PyDType>,
        priority: Option<PyPriority>,
        recomputable: bool,
        pinned: bool,
    ) -> Self {
        Self {
            inner: ExecutionState {
                id,
                state_type: state_type.inner,
                name,
                location: location.map(|l| l.inner).unwrap_or(Location::NotMaterialized),
                size_bytes,
                dtype: dtype.map(|d| d.inner).unwrap_or(DType::Float16),
                shape,
                produced_by: None,
                consumed_by: Vec::new(),
                depends_on: Vec::new(),
                compute_cost_ms: None,
                priority: priority.map(|p| p.inner).unwrap_or(Priority::Normal),
                pinned,
                recomputable,
                approximations: Vec::new(),
                current_approximation: None,
                metadata: Default::default(),
            },
        }
    }

    /// Create a KV cache state.
    #[staticmethod]
    #[pyo3(signature = (layer, size_bytes, shape, batch_size = 1, seq_len = 2048))]
    fn kv_cache(
        layer: usize,
        size_bytes: u64,
        shape: Vec<usize>,
        batch_size: usize,
        seq_len: usize,
    ) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Self {
            inner: ExecutionState::kv_cache(
                id,
                layer,
                batch_size,
                seq_len,
                32,  // default num_kv_heads
                128, // default head_dim
            ),
        }
    }

    /// Create an activation state.
    #[staticmethod]
    #[pyo3(signature = (layer, size_bytes, shape, batch_size = 1, seq_len = 2048))]
    fn activation(
        layer: usize,
        size_bytes: u64,
        shape: Vec<usize>,
        batch_size: usize,
        seq_len: usize,
    ) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Self {
            inner: ExecutionState::activation(
                id,
                layer,
                batch_size,
                seq_len,
                4096, // default hidden_dim
            ),
        }
    }

    /// Create an expert weight state.
    #[staticmethod]
    fn expert_weight(layer: usize, expert: usize, size_bytes: u64, shape: Vec<usize>) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Self {
            inner: ExecutionState::expert_weight(id, layer, expert, size_bytes),
        }
    }

    // Properties
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn size_bytes(&self) -> u64 {
        self.inner.size_bytes
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn location(&self) -> PyLocation {
        PyLocation { inner: self.inner.location }
    }

    #[getter]
    fn state_type(&self) -> PyStateType {
        PyStateType { inner: self.inner.state_type.clone() }
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType { inner: self.inner.dtype }
    }

    #[getter]
    fn priority(&self) -> PyPriority {
        PyPriority { inner: self.inner.priority }
    }

    #[getter]
    fn recomputable(&self) -> bool {
        self.inner.recomputable
    }

    #[getter]
    fn pinned(&self) -> bool {
        self.inner.pinned
    }

    #[getter]
    fn compute_cost_ms(&self) -> Option<f64> {
        self.inner.compute_cost_ms
    }

    // Setters
    #[setter]
    fn set_location(&mut self, location: PyLocation) {
        self.inner.location = location.inner;
    }

    #[setter]
    fn set_priority(&mut self, priority: PyPriority) {
        self.inner.priority = priority.inner;
    }

    #[setter]
    fn set_pinned(&mut self, pinned: bool) {
        self.inner.pinned = pinned;
    }

    #[setter]
    fn set_recomputable(&mut self, recomputable: bool) {
        self.inner.recomputable = recomputable;
    }

    #[setter]
    fn set_compute_cost_ms(&mut self, cost: Option<f64>) {
        self.inner.compute_cost_ms = cost;
    }

    // Methods
    /// Add a dependency on another state.
    fn add_dependency(&mut self, state_id: u64) {
        self.inner.depends_on.push(state_id);
    }

    /// Add a consumer operation.
    fn add_consumer(&mut self, operation_id: u64) {
        self.inner.consumed_by.push(operation_id);
    }

    /// Set the producing operation.
    fn set_producer(&mut self, operation_id: u64) {
        self.inner.produced_by = Some(operation_id);
    }

    /// Get size in human-readable format.
    fn size_human(&self) -> String {
        let bytes = self.inner.size_bytes;
        if bytes >= 1024 * 1024 * 1024 {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else if bytes >= 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} B", bytes)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "State(id={}, name='{}', type={:?}, location={:?}, size={})",
            self.inner.id,
            self.inner.name,
            self.inner.state_type,
            self.inner.location,
            self.size_human()
        )
    }
}
