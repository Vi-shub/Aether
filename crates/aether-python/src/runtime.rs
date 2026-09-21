//! Python wrapper for Aether runtime.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;

use aether_core::{
    CostModel, DecisionEngine, ExecutionGraph, HardwareConfig, StateManager,
    TransformerGraphBuilder,
};

use crate::{
    PyExecutionPlan, PyHardwareConfig, PyState, PyStateDecision,
};

/// Runtime configuration.
#[pyclass(name = "RuntimeConfig")]
#[derive(Clone)]
pub struct PyRuntimeConfig {
    /// Hardware configuration.
    pub hardware: PyHardwareConfig,
    /// Enable prefetching.
    pub enable_prefetch: bool,
    /// Enable recomputation.
    pub enable_recompute: bool,
    /// Memory pressure threshold (0.0-1.0).
    pub memory_pressure_threshold: f64,
    /// Prefetch lookahead operations.
    pub prefetch_lookahead: usize,
}

#[pymethods]
impl PyRuntimeConfig {
    /// Create a new runtime configuration.
    #[new]
    #[pyo3(signature = (
        hardware = None,
        enable_prefetch = true,
        enable_recompute = true,
        memory_pressure_threshold = 0.85,
        prefetch_lookahead = 4
    ))]
    fn new(
        hardware: Option<PyHardwareConfig>,
        enable_prefetch: bool,
        enable_recompute: bool,
        memory_pressure_threshold: f64,
        prefetch_lookahead: usize,
    ) -> Self {
        Self {
            hardware: hardware.unwrap_or_else(PyHardwareConfig::h100),
            enable_prefetch,
            enable_recompute,
            memory_pressure_threshold,
            prefetch_lookahead,
        }
    }

    /// Create H100 config.
    #[staticmethod]
    fn h100() -> Self {
        Self {
            hardware: PyHardwareConfig::h100(),
            enable_prefetch: true,
            enable_recompute: true,
            memory_pressure_threshold: 0.85,
            prefetch_lookahead: 4,
        }
    }

    /// Create A100 config.
    #[staticmethod]
    fn a100() -> Self {
        Self {
            hardware: PyHardwareConfig::a100(),
            enable_prefetch: true,
            enable_recompute: true,
            memory_pressure_threshold: 0.85,
            prefetch_lookahead: 4,
        }
    }

    /// Create memory-constrained config.
    #[staticmethod]
    fn memory_constrained(gpu_memory_gb: u64) -> Self {
        Self {
            hardware: PyHardwareConfig::custom(gpu_memory_gb, 256),
            enable_prefetch: true,
            enable_recompute: true,
            memory_pressure_threshold: 0.75,
            prefetch_lookahead: 2,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RuntimeConfig(hardware={}, prefetch={}, recompute={})",
            self.hardware.__repr__(),
            self.enable_prefetch,
            self.enable_recompute
        )
    }
}

/// Runtime metrics.
#[pyclass(name = "RuntimeMetrics")]
#[derive(Clone, Default)]
pub struct PyRuntimeMetrics {
    pub gpu_memory_used: u64,
    pub gpu_memory_available: u64,
    pub cpu_memory_used: u64,
    pub states_on_gpu: usize,
    pub states_on_cpu: usize,
    pub states_on_nvme: usize,
    pub pending_transfers: usize,
    pub operations_executed: u64,
    pub moves_executed: u64,
    pub recomputes_executed: u64,
    pub cache_hit_rate: f64,
}

#[pymethods]
impl PyRuntimeMetrics {
    #[getter]
    fn gpu_memory_used(&self) -> u64 {
        self.gpu_memory_used
    }

    #[getter]
    fn gpu_memory_available(&self) -> u64 {
        self.gpu_memory_available
    }

    #[getter]
    fn gpu_memory_utilization(&self) -> f64 {
        let total = self.gpu_memory_used + self.gpu_memory_available;
        if total > 0 {
            self.gpu_memory_used as f64 / total as f64
        } else {
            0.0
        }
    }

    #[getter]
    fn total_states(&self) -> usize {
        self.states_on_gpu + self.states_on_cpu + self.states_on_nvme
    }

    fn __repr__(&self) -> String {
        format!(
            "RuntimeMetrics(GPU: {:.1}% used, {} states)",
            self.gpu_memory_utilization() * 100.0,
            self.total_states()
        )
    }
}

/// The Aether runtime.
#[pyclass(name = "Runtime")]
pub struct PyRuntime {
    config: PyRuntimeConfig,
    state_manager: Arc<StateManager>,
    decision_engine: Arc<DecisionEngine>,
    cost_model: Arc<CostModel>,
    execution_graph: Option<ExecutionGraph>,
}

#[pymethods]
impl PyRuntime {
    /// Create a new runtime with configuration.
    #[new]
    fn new(config: Option<PyRuntimeConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_else(PyRuntimeConfig::h100);
        let hardware = config.hardware.inner.clone();

        let state_manager = Arc::new(StateManager::new(hardware.clone()));
        let cost_model = Arc::new(CostModel::new(hardware.clone()));
        let decision_engine = Arc::new(DecisionEngine::new(cost_model.clone(), hardware));

        Ok(Self {
            config,
            state_manager,
            decision_engine,
            cost_model,
            execution_graph: None,
        })
    }

    /// Create H100 runtime.
    #[staticmethod]
    fn h100() -> PyResult<Self> {
        Self::new(Some(PyRuntimeConfig::h100()))
    }

    /// Create A100 runtime.
    #[staticmethod]
    fn a100() -> PyResult<Self> {
        Self::new(Some(PyRuntimeConfig::a100()))
    }

    /// Create memory-constrained runtime.
    #[staticmethod]
    fn memory_constrained(gpu_memory_gb: u64) -> PyResult<Self> {
        Self::new(Some(PyRuntimeConfig::memory_constrained(gpu_memory_gb)))
    }

    /// Register a state with the runtime.
    fn register_state(&self, state: PyState) -> PyResult<()> {
        self.state_manager.register(state.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Register multiple states.
    fn register_states(&self, states: Vec<PyState>) -> PyResult<()> {
        for state in states {
            self.register_state(state)?;
        }
        Ok(())
    }

    /// Unregister a state.
    fn unregister_state(&self, state_id: u64) -> bool {
        self.state_manager.unregister(state_id)
    }

    /// Get a state by ID.
    fn get_state(&self, state_id: u64) -> Option<PyState> {
        self.state_manager.get_state(state_id)
            .map(|s| PyState { inner: s })
    }

    /// Get all states.
    fn get_all_states(&self) -> Vec<PyState> {
        self.state_manager.all_states()
            .into_iter()
            .map(|s| PyState { inner: s })
            .collect()
    }

    /// Get states on GPU.
    fn get_states_on_gpu(&self) -> Vec<PyState> {
        self.state_manager.states_on_gpu()
            .into_iter()
            .map(|s| PyState { inner: s })
            .collect()
    }

    /// Get total memory used on GPU.
    fn gpu_memory_used(&self) -> u64 {
        self.state_manager.memory_used(aether_core::Location::Hbm)
    }

    /// Build an execution graph for a transformer.
    fn build_transformer_graph(
        &mut self,
        num_layers: usize,
        hidden_dim: usize,
        batch_size: usize,
        seq_len: usize,
        is_prefill: bool,
        is_moe: bool,
    ) -> PyResult<()> {
        let builder = TransformerGraphBuilder::new(num_layers, hidden_dim);

        let graph = if is_moe {
            builder.build_moe_prefill(batch_size, seq_len, 8, 2)
        } else if is_prefill {
            builder.build_prefill(batch_size, seq_len, false)
        } else {
            builder.build_decode(batch_size, seq_len)
        };

        self.execution_graph = Some(graph);
        Ok(())
    }

    /// Plan execution.
    fn plan(&self) -> PyResult<PyExecutionPlan> {
        let states = self.state_manager.all_states();

        // Use a default graph if none set
        let graph = self.execution_graph.clone()
            .unwrap_or_else(|| {
                let builder = TransformerGraphBuilder::new(32, 4096);
                builder.build_prefill(1, 512, false)
            });

        let plan = self.decision_engine.plan(&states, &graph);

        Ok(PyExecutionPlan { inner: plan })
    }

    /// Execute a plan (simulation).
    fn execute(&self, plan: PyExecutionPlan) -> PyResult<()> {
        // In production, this would execute the plan
        // For now, just update state locations
        for decision in plan.inner.iter() {
            if let Some(target) = decision.target_location {
                self.state_manager.update_location(decision.state_id, target);
            }
        }
        Ok(())
    }

    /// Get runtime metrics.
    fn metrics(&self) -> PyRuntimeMetrics {
        let all_states = self.state_manager.all_states();

        let states_on_gpu = all_states.iter()
            .filter(|s| s.location == aether_core::Location::Hbm)
            .count();
        let states_on_cpu = all_states.iter()
            .filter(|s| s.location == aether_core::Location::Dram)
            .count();
        let states_on_nvme = all_states.iter()
            .filter(|s| s.location == aether_core::Location::Nvme)
            .count();

        let gpu_used = self.state_manager.memory_used(aether_core::Location::Hbm);
        let gpu_total = self.config.hardware.inner.hbm.capacity_bytes;

        PyRuntimeMetrics {
            gpu_memory_used: gpu_used,
            gpu_memory_available: gpu_total.saturating_sub(gpu_used),
            cpu_memory_used: self.state_manager.memory_used(aether_core::Location::Dram),
            states_on_gpu,
            states_on_cpu,
            states_on_nvme,
            pending_transfers: 0,
            operations_executed: 0,
            moves_executed: 0,
            recomputes_executed: 0,
            cache_hit_rate: 1.0,
        }
    }

    /// Estimate cost to move a state.
    fn estimate_move_cost(&self, state: PyState, from_location: &str, to_location: &str) -> f64 {
        let from = match from_location {
            "hbm" | "gpu" => aether_core::Location::Hbm,
            "dram" | "cpu" => aether_core::Location::Dram,
            "nvme" | "ssd" => aether_core::Location::Nvme,
            _ => aether_core::Location::Hbm,
        };

        let to = match to_location {
            "hbm" | "gpu" => aether_core::Location::Hbm,
            "dram" | "cpu" => aether_core::Location::Dram,
            "nvme" | "ssd" => aether_core::Location::Nvme,
            _ => aether_core::Location::Dram,
        };

        self.cost_model.move_cost(&state.inner, from, to).time_ms
    }

    /// Compare move vs recompute cost.
    fn compare_move_vs_recompute(
        &self,
        state: PyState,
        from_location: &str,
        to_location: &str,
    ) -> PyObject {
        Python::with_gil(|py| {
            let from = match from_location {
                "hbm" | "gpu" => aether_core::Location::Hbm,
                "dram" | "cpu" => aether_core::Location::Dram,
                "nvme" | "ssd" => aether_core::Location::Nvme,
                _ => aether_core::Location::Hbm,
            };

            let to = match to_location {
                "hbm" | "gpu" => aether_core::Location::Hbm,
                "dram" | "cpu" => aether_core::Location::Dram,
                "nvme" | "ssd" => aether_core::Location::Nvme,
                _ => aether_core::Location::Dram,
            };

            let comparison = self.cost_model.compare_move_vs_recompute(&state.inner, from, to);

            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("move_time_ms", comparison.move_time_ms).unwrap();
            dict.set_item("recompute_time_ms", comparison.recompute_time_ms).unwrap();
            dict.set_item("should_recompute", comparison.should_recompute).unwrap();
            dict.set_item("savings_ms", comparison.savings_ms).unwrap();

            dict.into()
        })
    }

    /// Get configuration.
    #[getter]
    fn config(&self) -> PyRuntimeConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        let states = self.state_manager.all_states().len();
        let gpu_used = self.gpu_memory_used();
        format!(
            "Runtime(states={}, gpu_used={:.1}GB, config={})",
            states,
            gpu_used as f64 / 1e9,
            self.config.hardware.__repr__()
        )
    }
}
