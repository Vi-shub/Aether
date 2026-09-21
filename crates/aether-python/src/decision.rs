//! Python wrappers for decision types.

use pyo3::prelude::*;
use aether_core::{StateDecision, ExecutionPlan, Action, Location};

use crate::types::{PyAction, PyLocation};

/// A decision about what to do with a state.
#[pyclass(name = "StateDecision")]
#[derive(Clone)]
pub struct PyStateDecision {
    pub inner: StateDecision,
}

#[pymethods]
impl PyStateDecision {
    /// Create a new state decision.
    #[new]
    fn new(
        state_id: u64,
        action: PyAction,
        target_location: Option<PyLocation>,
        estimated_time_ms: f64,
        estimated_memory_delta: i64,
    ) -> Self {
        Self {
            inner: StateDecision {
                state_id,
                action: action.inner,
                target_location: target_location.map(|l| l.inner),
                estimated_time_ms,
                estimated_memory_delta,
                reason: String::new(),
            },
        }
    }

    #[getter]
    fn state_id(&self) -> u64 {
        self.inner.state_id
    }

    #[getter]
    fn action(&self) -> PyAction {
        PyAction { inner: self.inner.action }
    }

    #[getter]
    fn target_location(&self) -> Option<PyLocation> {
        self.inner.target_location.map(|l| PyLocation { inner: l })
    }

    #[getter]
    fn estimated_time_ms(&self) -> f64 {
        self.inner.estimated_time_ms
    }

    #[getter]
    fn estimated_memory_delta(&self) -> i64 {
        self.inner.estimated_memory_delta
    }

    #[getter]
    fn reason(&self) -> String {
        self.inner.reason.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "StateDecision(state={}, action={:?}, target={:?}, time={:.2}ms)",
            self.inner.state_id,
            self.inner.action,
            self.inner.target_location,
            self.inner.estimated_time_ms
        )
    }
}

/// A plan containing multiple state decisions.
#[pyclass(name = "ExecutionPlan")]
#[derive(Clone)]
pub struct PyExecutionPlan {
    pub inner: ExecutionPlan,
}

#[pymethods]
impl PyExecutionPlan {
    /// Create an empty execution plan.
    #[new]
    fn new() -> Self {
        Self {
            inner: ExecutionPlan::new(),
        }
    }

    /// Add a decision to the plan.
    fn add_decision(&mut self, decision: PyStateDecision) {
        self.inner.add_decision(decision.inner);
    }

    /// Get number of decisions.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if plan is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get total estimated time.
    fn total_time_ms(&self) -> f64 {
        self.inner.total_time_ms()
    }

    /// Get total memory delta.
    fn total_memory_delta(&self) -> i64 {
        self.inner.total_memory_delta()
    }

    /// Get all decisions.
    fn decisions(&self) -> Vec<PyStateDecision> {
        self.inner.iter()
            .map(|d| PyStateDecision { inner: d.clone() })
            .collect()
    }

    /// Get decisions by action type.
    fn decisions_by_action(&self, action: PyAction) -> Vec<PyStateDecision> {
        self.inner.iter()
            .filter(|d| d.action == action.inner)
            .map(|d| PyStateDecision { inner: d.clone() })
            .collect()
    }

    /// Count decisions by action type.
    fn count_actions(&self) -> PyObject {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);

            let mut keep = 0;
            let mut move_ = 0;
            let mut prefetch = 0;
            let mut recompute = 0;
            let mut evict = 0;
            let mut other = 0;

            for decision in self.inner.iter() {
                match decision.action {
                    Action::Keep => keep += 1,
                    Action::Move => move_ += 1,
                    Action::Prefetch => prefetch += 1,
                    Action::Recompute => recompute += 1,
                    Action::Evict => evict += 1,
                    _ => other += 1,
                }
            }

            dict.set_item("keep", keep).unwrap();
            dict.set_item("move", move_).unwrap();
            dict.set_item("prefetch", prefetch).unwrap();
            dict.set_item("recompute", recompute).unwrap();
            dict.set_item("evict", evict).unwrap();
            dict.set_item("other", other).unwrap();

            dict.into()
        })
    }

    /// Get a summary of the plan.
    fn summary(&self) -> String {
        format!(
            "ExecutionPlan: {} decisions, {:.2}ms estimated, {} bytes memory delta",
            self.inner.len(),
            self.inner.total_time_ms(),
            self.inner.total_memory_delta()
        )
    }

    fn __repr__(&self) -> String {
        self.summary()
    }

    fn __iter__(slf: PyRef<Self>) -> PyResult<PyExecutionPlanIter> {
        Ok(PyExecutionPlanIter {
            inner: slf.inner.iter().cloned().collect(),
            index: 0,
        })
    }
}

/// Iterator for ExecutionPlan.
#[pyclass]
pub struct PyExecutionPlanIter {
    inner: Vec<StateDecision>,
    index: usize,
}

#[pymethods]
impl PyExecutionPlanIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyStateDecision> {
        if slf.index < slf.inner.len() {
            let decision = slf.inner[slf.index].clone();
            slf.index += 1;
            Some(PyStateDecision { inner: decision })
        } else {
            None
        }
    }
}
