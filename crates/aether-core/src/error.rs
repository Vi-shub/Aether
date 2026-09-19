//! Error types for Aether runtime.

use thiserror::Error;
use crate::types::{StateId, Location};

/// Result type for Aether operations.
pub type Result<T> = std::result::Result<T, AetherError>;

/// Errors that can occur in Aether operations.
#[derive(Error, Debug)]
pub enum AetherError {
    /// State not found in registry.
    #[error("State not found: {0}")]
    StateNotFound(StateId),

    /// Insufficient memory at target location.
    #[error("Insufficient memory at {location:?}: need {required} bytes, have {available} bytes")]
    InsufficientMemory {
        location: Location,
        required: u64,
        available: u64,
    },

    /// Invalid state transition.
    #[error("Invalid state transition: cannot move from {from:?} to {to:?}")]
    InvalidTransition {
        from: Location,
        to: Location,
    },

    /// State is pinned and cannot be evicted.
    #[error("State {0} is pinned and cannot be evicted")]
    StatePinned(StateId),

    /// State cannot be recomputed.
    #[error("State {0} cannot be recomputed (no producer or non-recomputable type)")]
    NotRecomputable(StateId),

    /// Transfer failed.
    #[error("Transfer failed: {0}")]
    TransferFailed(String),

    /// Planning failed.
    #[error("Planning failed: {0}")]
    PlanningFailed(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Operation timed out.
    #[error("Operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// CUDA error.
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl AetherError {
    /// Create a new internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Create a new planning error.
    pub fn planning(msg: impl Into<String>) -> Self {
        Self::PlanningFailed(msg.into())
    }

    /// Create a new transfer error.
    pub fn transfer(msg: impl Into<String>) -> Self {
        Self::TransferFailed(msg.into())
    }
}

// Implement From for common error types
impl From<std::io::Error> for AetherError {
    fn from(err: std::io::Error) -> Self {
        Self::Internal(err.to_string())
    }
}

impl From<serde_json::Error> for AetherError {
    fn from(err: serde_json::Error) -> Self {
        Self::Internal(err.to_string())
    }
}
