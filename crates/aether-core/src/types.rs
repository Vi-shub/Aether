//! Core type definitions for Aether runtime.
//!
//! This module defines the fundamental types used throughout Aether:
//! - [`StateType`]: What kind of state (KV, activation, expert, etc.)
//! - [`Location`]: Where state lives (HBM, DRAM, NVMe, etc.)
//! - [`Action`]: What to do with state (keep, move, recompute, etc.)

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Identity Types
// =============================================================================

/// Unique identifier for a piece of execution state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateId(pub String);

impl StateId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for StateId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for StateId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Unique identifier for an operation in the execution graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationId(pub String);

impl OperationId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Layer index in a model.
pub type LayerId = u32;

/// Expert index in an MoE model.
pub type ExpertId = u32;

/// Request identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub String);

// =============================================================================
// State Types
// =============================================================================

/// Classification of execution state in AI inference.
///
/// Aether treats all state uniformly but with type-specific cost models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum StateType {
    // Model state (typically static during inference)
    /// Dense layer weights.
    Weight = 0,
    /// MoE expert weights - can be loaded/evicted dynamically.
    ExpertWeight = 1,
    /// Embedding tables.
    Embedding = 2,

    // Request state (generated during inference)
    /// Key-value cache for attention.
    KvCache = 10,
    /// Layer activations - can be checkpointed/recomputed.
    Activation = 11,
    /// Intermediate tensors within an operation.
    Intermediate = 12,
    /// Attention masks, position encodings, etc.
    AttentionMetadata = 13,

    // Execution state (runtime decisions)
    /// Draft tokens and speculative execution state.
    SpeculativeState = 20,
    /// MoE routing decisions for potential reuse.
    RoutingDecision = 21,

    // Application state (higher-level)
    /// Agent memory, tool results, session state.
    AgentContext = 30,
    /// RAG/retrieval results.
    RetrievedState = 31,
}

impl StateType {
    /// Check if this is a model weight type.
    pub fn is_weight(&self) -> bool {
        matches!(self, Self::Weight | Self::ExpertWeight | Self::Embedding)
    }

    /// Check if this is recomputable (not a weight).
    pub fn is_recomputable(&self) -> bool {
        !self.is_weight()
    }
}

// =============================================================================
// Memory Locations
// =============================================================================

/// Memory tier where state can reside.
///
/// Ordered roughly by access latency (fastest to slowest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Location {
    // GPU memory
    /// GPU High Bandwidth Memory - fastest, most limited.
    Hbm = 0,

    // CPU memory
    /// CPU DRAM - larger, slower than HBM.
    Dram = 1,

    // Extended memory
    /// CXL-attached memory - capacity tier.
    Cxl = 2,

    // Storage
    /// NVMe SSD - large capacity, higher latency.
    Nvme = 3,

    // Remote
    /// Another GPU's HBM (via NVLink/PCIe/RDMA).
    RemoteGpu = 10,
    /// Another host's memory (via RDMA/network).
    RemoteHost = 11,

    // Virtual states
    /// State exists logically but hasn't been computed yet.
    NotMaterialized = 20,
    /// State was computed but has been discarded.
    Evicted = 21,
}

impl Location {
    /// Check if this is a local memory tier.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Hbm | Self::Dram | Self::Cxl | Self::Nvme)
    }

    /// Check if this is GPU memory.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Hbm | Self::RemoteGpu)
    }

    /// Check if this requires network access.
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::RemoteGpu | Self::RemoteHost)
    }

    /// Check if state actually exists in memory.
    pub fn is_materialized(&self) -> bool {
        !matches!(self, Self::NotMaterialized | Self::Evicted)
    }

    /// Get typical bandwidth in GB/s for this tier (read).
    pub fn typical_bandwidth_gbps(&self) -> f64 {
        match self {
            Self::Hbm => 2000.0,        // A100: ~2TB/s
            Self::Dram => 100.0,        // DDR5
            Self::Cxl => 50.0,          // CXL 2.0
            Self::Nvme => 7.0,          // NVMe Gen4
            Self::RemoteGpu => 300.0,   // NVLink
            Self::RemoteHost => 12.5,   // 100Gbps
            Self::NotMaterialized => 0.0,
            Self::Evicted => 0.0,
        }
    }

    /// Get typical latency in microseconds.
    pub fn typical_latency_us(&self) -> f64 {
        match self {
            Self::Hbm => 0.1,
            Self::Dram => 0.1,
            Self::Cxl => 0.3,
            Self::Nvme => 10.0,
            Self::RemoteGpu => 1.0,
            Self::RemoteHost => 5.0,
            Self::NotMaterialized => 0.0,
            Self::Evicted => 0.0,
        }
    }
}

// =============================================================================
// Actions
// =============================================================================

/// Actions the decision engine can take for a piece of state.
///
/// This is the core of Aether: choosing between these options based on
/// the execution graph, cost model, and constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Action {
    /// Keep state in current location.
    ///
    /// Use when: State is already where it needs to be.
    /// Cost: 0 (no action needed)
    Keep = 0,

    /// Transfer state to a different memory tier.
    ///
    /// Use when: State needed in faster tier, or evicting to slower tier.
    /// Cost: transfer_time = size / bandwidth
    Move = 1,

    /// Proactively move state before it's needed.
    ///
    /// Use when: Future operation will need this state.
    /// Cost: Same as MOVE, but overlapped with other computation.
    Prefetch = 2,

    /// Discard state and recompute when needed.
    ///
    /// Use when: recompute_cost < move_cost, or memory pressure is high.
    /// Cost: compute_time (depends on operation)
    Recompute = 3,

    /// Use a lower-fidelity version of the state.
    ///
    /// Use when: Approximate is good enough and saves memory/compute.
    /// Options: quantization, sparsification, compression.
    Approximate = 4,

    /// Create a copy in another location.
    ///
    /// Use when: State needed in multiple places simultaneously.
    /// Cost: transfer_time + memory_cost
    Replicate = 5,

    /// Remove state from memory.
    ///
    /// Use when: Memory pressure is high and state won't be needed soon.
    /// Cost: 0 (but may incur future recompute/reload cost)
    Evict = 6,

    /// Defer the decision to a later time.
    ///
    /// Use when: Insufficient information to decide, or waiting is optimal.
    Delay = 7,
}

impl Action {
    /// Check if this is a movement action.
    pub fn is_movement(&self) -> bool {
        matches!(self, Self::Move | Self::Prefetch | Self::Replicate)
    }

    /// Check if this involves computation.
    pub fn is_compute(&self) -> bool {
        matches!(self, Self::Recompute | Self::Approximate)
    }
}

// =============================================================================
// Operation Types
// =============================================================================

/// Types of operations in the execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OperationType {
    // Attention operations
    Attention = 0,
    PrefillAttention = 1,
    DecodeAttention = 2,

    // Linear operations
    Linear = 10,
    QkvProj = 11,
    OutputProj = 12,

    // MLP operations
    Mlp = 20,
    GateProj = 21,
    UpProj = 22,
    DownProj = 23,

    // MoE operations
    Router = 30,
    Expert = 31,
    ExpertCombine = 32,

    // Other
    Embedding = 40,
    LayerNorm = 41,
    RmsNorm = 42,
    Sampling = 43,

    // Meta operations
    Transfer = 50,
    Sync = 51,
}

// =============================================================================
// Approximation Methods
// =============================================================================

/// Methods for approximating state to save memory/compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ApproximationMethod {
    // Quantization
    Int8 = 0,
    Int4 = 1,
    Fp8 = 2,
    Fp16 = 3,

    // Sparsification
    TopK = 10,
    Threshold = 11,
    StructuredSparse = 12,

    // Compression
    Zstd = 20,
    Lz4 = 21,

    // KV-specific
    KvQuantize = 30,
    KvCompress = 31,
    SlidingWindow = 32,
}

// =============================================================================
// Priority Levels
// =============================================================================

/// Priority levels for state/requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Priority {
    /// Must not be evicted - actively in use.
    Critical = 0,
    /// Important - will be needed very soon.
    High = 1,
    /// Standard priority.
    Normal = 2,
    /// Can be evicted if needed.
    Low = 3,
    /// Speculative/prefetch - evict first.
    Background = 4,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

// =============================================================================
// Data Types
// =============================================================================

/// Tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DType {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Float8E4M3 = 3,
    Float8E5M2 = 4,
    Int8 = 10,
    Int4 = 11,
    Uint8 = 12,
}

impl DType {
    /// Size in bytes per element.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Float8E4M3 | Self::Float8E5M2 | Self::Int8 | Self::Uint8 => 1,
            Self::Int4 => 1, // Packed, but we round up
        }
    }
}

impl Default for DType {
    fn default() -> Self {
        Self::Float16
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_type_recomputable() {
        assert!(!StateType::Weight.is_recomputable());
        assert!(!StateType::ExpertWeight.is_recomputable());
        assert!(StateType::KvCache.is_recomputable());
        assert!(StateType::Activation.is_recomputable());
    }

    #[test]
    fn test_location_properties() {
        assert!(Location::Hbm.is_gpu());
        assert!(Location::Hbm.is_materialized());
        assert!(!Location::Dram.is_gpu());
        assert!(Location::Dram.is_local());
        assert!(!Location::NotMaterialized.is_materialized());
        assert!(Location::RemoteGpu.is_remote());
    }

    #[test]
    fn test_action_properties() {
        assert!(Action::Move.is_movement());
        assert!(Action::Prefetch.is_movement());
        assert!(!Action::Keep.is_movement());
        assert!(Action::Recompute.is_compute());
        assert!(!Action::Move.is_compute());
    }

    #[test]
    fn test_state_id_from() {
        let id: StateId = "test".into();
        assert_eq!(id.0, "test");
        
        let id: StateId = String::from("test2").into();
        assert_eq!(id.0, "test2");
    }
}
