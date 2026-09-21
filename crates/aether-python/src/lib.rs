//! # Aether Python Bindings
//!
//! Python interface to the Aether AI Execution Runtime.
//!
//! ## Example
//!
//! ```python
//! import aether
//!
//! # Create runtime
//! runtime = aether.Runtime.h100()
//!
//! # Create state
//! kv_cache = aether.State.kv_cache(
//!     layer=0,
//!     size_bytes=1024*1024*128,  # 128 MB
//!     shape=[1, 2048, 32, 128]
//! )
//!
//! # Register state
//! runtime.register_state(kv_cache)
//!
//! # Plan execution
//! plan = runtime.plan()
//!
//! # Execute plan
//! runtime.execute(plan)
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::sync::Arc;

mod types;
mod state;
mod decision;
mod runtime;
mod hardware;

pub use types::*;
pub use state::*;
pub use decision::*;
pub use runtime::*;
pub use hardware::*;

/// Aether AI Execution Runtime
///
/// A state-aware runtime for AI inference that jointly optimizes
/// computation and state movement.
#[pymodule]
fn aether(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "Aether AI Execution Runtime")?;

    // Core types
    m.add_class::<PyStateType>()?;
    m.add_class::<PyLocation>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyDType>()?;
    m.add_class::<PyPriority>()?;

    // State management
    m.add_class::<PyState>()?;
    m.add_class::<PyStateDecision>()?;
    m.add_class::<PyExecutionPlan>()?;

    // Hardware configuration
    m.add_class::<PyHardwareConfig>()?;
    m.add_class::<PyMemoryTier>()?;

    // Runtime
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyRuntimeConfig>()?;
    m.add_class::<PyRuntimeMetrics>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(get_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_kv_cache_size, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_model_size, m)?)?;

    Ok(())
}

// =============================================================================
// Utility functions
// =============================================================================

/// Get the number of available CUDA devices.
#[pyfunction]
fn get_device_count() -> PyResult<usize> {
    // In production, this would query CUDA
    Ok(1) // Mock
}

/// Get information about a CUDA device.
#[pyfunction]
fn get_device_info(device_id: i32) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("name", "NVIDIA GPU (mock)")?;
        dict.set_item("memory_gb", 80.0)?;
        dict.set_item("compute_capability", "9.0")?;
        dict.set_item("device_id", device_id)?;
        Ok(dict.into())
    })
}

/// Estimate KV cache size for a model configuration.
///
/// Args:
///     num_layers: Number of transformer layers
///     num_kv_heads: Number of key-value heads
///     head_dim: Dimension per head
///     batch_size: Batch size
///     seq_len: Sequence length
///     dtype: Data type (default: "float16")
///
/// Returns:
///     Size in bytes
#[pyfunction]
#[pyo3(signature = (num_layers, num_kv_heads, head_dim, batch_size, seq_len, dtype="float16"))]
fn estimate_kv_cache_size(
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    batch_size: usize,
    seq_len: usize,
    dtype: &str,
) -> PyResult<u64> {
    let dtype_size = match dtype {
        "float16" | "bfloat16" => 2,
        "float32" => 4,
        "int8" => 1,
        _ => return Err(PyValueError::new_err(format!("Unknown dtype: {}", dtype))),
    };

    // 2 (K+V) * layers * batch * seq * heads * head_dim * dtype_size
    let size = 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * dtype_size;
    Ok(size as u64)
}

/// Estimate model size in bytes.
///
/// Args:
///     num_layers: Number of transformer layers
///     hidden_dim: Hidden dimension
///     intermediate_dim: FFN intermediate dimension
///     vocab_size: Vocabulary size
///     num_experts: Number of MoE experts (0 for dense models)
///     dtype: Data type (default: "float16")
///
/// Returns:
///     Size in bytes
#[pyfunction]
#[pyo3(signature = (num_layers, hidden_dim, intermediate_dim, vocab_size, num_experts=0, dtype="float16"))]
fn estimate_model_size(
    num_layers: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    vocab_size: usize,
    num_experts: usize,
    dtype: &str,
) -> PyResult<u64> {
    let dtype_size = match dtype {
        "float16" | "bfloat16" => 2,
        "float32" => 4,
        "int8" => 1,
        _ => return Err(PyValueError::new_err(format!("Unknown dtype: {}", dtype))),
    };

    let embedding = vocab_size * hidden_dim * dtype_size;

    let per_layer = if num_experts > 0 {
        // MoE model
        let attention = 4 * hidden_dim * hidden_dim * dtype_size;
        let expert_ffn = 3 * hidden_dim * intermediate_dim * dtype_size;
        let router = hidden_dim * num_experts * dtype_size;
        attention + (expert_ffn * num_experts) + router
    } else {
        // Dense model
        let attention = 4 * hidden_dim * hidden_dim * dtype_size;
        let ffn = 3 * hidden_dim * intermediate_dim * dtype_size;
        attention + ffn
    };

    let total = embedding + num_layers * per_layer;
    Ok(total as u64)
}
