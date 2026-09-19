//! ExecutionState: The fundamental unit of state in Aether.
//!
//! Every piece of data generated or required during inference is an ExecutionState
//! with identity, location, dependencies, and cost information.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::types::*;

// =============================================================================
// State Counter for ID Generation
// =============================================================================

static STATE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_state_id(prefix: &str) -> StateId {
    let count = STATE_COUNTER.fetch_add(1, Ordering::Relaxed);
    StateId::new(format!("{}_{:08x}", prefix, count))
}

// =============================================================================
// Approximation Option
// =============================================================================

/// Configuration for an approximation method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproximationOption {
    /// The approximation technique.
    pub method: ApproximationMethod,
    /// Estimated quality degradation (0-1 scale).
    pub quality_loss: f64,
    /// Memory reduction factor (e.g., 0.5 = half size).
    pub size_reduction: f64,
    /// Time to perform approximation in milliseconds.
    pub compute_overhead_ms: f64,
}

// =============================================================================
// State Metadata
// =============================================================================

/// Additional metadata for execution state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMetadata {
    /// Timestamp when state was created (as duration since start).
    pub created_at: Duration,
    /// Timestamp of last access.
    pub last_accessed: Duration,
    /// Number of times this state has been accessed.
    pub access_count: u64,
    /// Predicted time of next access (if known).
    pub predicted_next_access: Option<Duration>,
    /// Probability this state will be accessed again (0-1).
    pub access_probability: f64,
    /// Request that generated this state.
    pub request_id: Option<RequestId>,
    /// Layer this state belongs to.
    pub layer_id: Option<LayerId>,
    /// Expert this state belongs to (for MoE).
    pub expert_id: Option<ExpertId>,
    /// Custom tags for filtering/grouping.
    pub tags: HashSet<String>,
}

impl Default for StateMetadata {
    fn default() -> Self {
        Self {
            created_at: Duration::ZERO,
            last_accessed: Duration::ZERO,
            access_count: 0,
            predicted_next_access: None,
            access_probability: 1.0,
            request_id: None,
            layer_id: None,
            expert_id: None,
            tags: HashSet::new(),
        }
    }
}

// =============================================================================
// ExecutionState
// =============================================================================

/// The fundamental unit of state in Aether.
///
/// Every piece of data generated or required during AI inference is
/// represented as an ExecutionState. This includes:
///
/// - Model weights and expert weights
/// - KV cache entries
/// - Activations and intermediate tensors
/// - Attention metadata
/// - Agent/application state
///
/// The key insight is that all state is treated uniformly for decision-making,
/// but with type-specific cost models and constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    // =========================================================================
    // Identity
    // =========================================================================
    /// Unique identifier for this state.
    pub id: StateId,
    /// What kind of state this is.
    pub state_type: StateType,
    /// Human-readable name for debugging.
    pub name: String,

    // =========================================================================
    // Physical Properties
    // =========================================================================
    /// Current memory location.
    pub location: Location,
    /// Memory footprint in bytes.
    pub size_bytes: u64,
    /// Tensor data type.
    pub dtype: DType,
    /// Tensor shape.
    pub shape: Vec<usize>,

    // =========================================================================
    // Dependencies
    // =========================================================================
    /// Operation that creates this state (None for model weights).
    pub produced_by: Option<OperationId>,
    /// Operations that read this state.
    pub consumed_by: Vec<OperationId>,
    /// Other states needed to recompute this state.
    pub depends_on: Vec<StateId>,

    // =========================================================================
    // Cost Information
    // =========================================================================
    /// Time to recompute this state from its dependencies (None if not recomputable).
    pub compute_cost_ms: Option<f64>,

    // =========================================================================
    // Scheduling
    // =========================================================================
    /// Eviction priority level.
    pub priority: Priority,
    /// If true, state cannot be evicted.
    pub pinned: bool,
    /// If true, state can be recomputed from dependencies.
    pub recomputable: bool,

    // =========================================================================
    // Approximation
    // =========================================================================
    /// Available approximation methods for this state.
    pub approximations: Vec<ApproximationOption>,
    /// Current approximation (None = full precision).
    pub current_approximation: Option<ApproximationMethod>,

    // =========================================================================
    // Metadata
    // =========================================================================
    /// Additional tracking information.
    pub metadata: StateMetadata,
}

impl ExecutionState {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a new ExecutionState with a generated ID.
    pub fn new(state_type: StateType, size_bytes: u64) -> Self {
        let prefix = match state_type {
            StateType::Weight => "weight",
            StateType::ExpertWeight => "expert",
            StateType::Embedding => "embed",
            StateType::KvCache => "kv",
            StateType::Activation => "act",
            StateType::Intermediate => "inter",
            StateType::AttentionMetadata => "attn_meta",
            StateType::SpeculativeState => "spec",
            StateType::RoutingDecision => "route",
            StateType::AgentContext => "agent",
            StateType::RetrievedState => "retrieved",
        };

        let id = generate_state_id(prefix);
        let name = id.0.clone();

        Self {
            id,
            state_type,
            name,
            location: Location::NotMaterialized,
            size_bytes,
            dtype: DType::Float16,
            shape: Vec::new(),
            produced_by: None,
            consumed_by: Vec::new(),
            depends_on: Vec::new(),
            compute_cost_ms: None,
            priority: Priority::Normal,
            pinned: false,
            recomputable: state_type.is_recomputable(),
            approximations: Vec::new(),
            current_approximation: None,
            metadata: StateMetadata::default(),
        }
    }

    /// Create a KV cache state.
    pub fn kv_cache(
        layer_id: LayerId,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Self {
        // KV cache: 2 (K and V) * seq_len * num_heads * head_dim * dtype_size
        let size_bytes = 2 * seq_len * num_heads * head_dim * dtype.size_bytes();

        let mut state = Self::new(StateType::KvCache, size_bytes as u64);
        state.name = format!("kv_layer_{}", layer_id);
        state.dtype = dtype;
        state.shape = vec![2, seq_len, num_heads, head_dim];
        state.metadata.layer_id = Some(layer_id);
        state.recomputable = true;
        state
    }

    /// Create an activation state.
    pub fn activation(
        layer_id: LayerId,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        dtype: DType,
    ) -> Self {
        let size_bytes = batch_size * seq_len * hidden_dim * dtype.size_bytes();

        let mut state = Self::new(StateType::Activation, size_bytes as u64);
        state.name = format!("activation_layer_{}", layer_id);
        state.dtype = dtype;
        state.shape = vec![batch_size, seq_len, hidden_dim];
        state.metadata.layer_id = Some(layer_id);
        state.recomputable = true;
        state
    }

    /// Create an MoE expert weight state.
    pub fn expert_weight(layer_id: LayerId, expert_id: ExpertId, size_bytes: u64) -> Self {
        let mut state = Self::new(StateType::ExpertWeight, size_bytes);
        state.name = format!("expert_L{}_E{}", layer_id, expert_id);
        state.metadata.layer_id = Some(layer_id);
        state.metadata.expert_id = Some(expert_id);
        state.recomputable = false; // Weights cannot be recomputed
        state
    }

    /// Create a weight state.
    pub fn weight(name: impl Into<String>, size_bytes: u64) -> Self {
        let mut state = Self::new(StateType::Weight, size_bytes);
        state.name = name.into();
        state.recomputable = false;
        state.location = Location::Hbm; // Weights typically start in HBM
        state
    }

    // =========================================================================
    // Builder Methods
    // =========================================================================

    /// Set the location.
    pub fn with_location(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    /// Set as pinned.
    pub fn with_pinned(mut self, pinned: bool) -> Self {
        self.pinned = pinned;
        self
    }

    /// Set compute cost.
    pub fn with_compute_cost(mut self, cost_ms: f64) -> Self {
        self.compute_cost_ms = Some(cost_ms);
        self
    }

    /// Set producer operation.
    pub fn with_producer(mut self, op_id: OperationId) -> Self {
        self.produced_by = Some(op_id);
        self
    }

    /// Add a consumer operation.
    pub fn with_consumer(mut self, op_id: OperationId) -> Self {
        self.consumed_by.push(op_id);
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /// Check if state exists in memory.
    #[inline]
    pub fn is_materialized(&self) -> bool {
        self.location.is_materialized()
    }

    /// Check if state is in GPU memory.
    #[inline]
    pub fn is_in_gpu(&self) -> bool {
        self.location.is_gpu()
    }

    /// Check if state can be recomputed.
    #[inline]
    pub fn can_recompute(&self) -> bool {
        self.recomputable && self.compute_cost_ms.is_some()
    }

    /// Size in megabytes.
    #[inline]
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Size in gigabytes.
    #[inline]
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    // =========================================================================
    // Methods
    // =========================================================================

    /// Update access tracking.
    pub fn touch(&mut self, now: Duration) {
        self.metadata.last_accessed = now;
        self.metadata.access_count += 1;
    }

    /// Calculate transfer time to a target location.
    pub fn transfer_time_ms(&self, source: Location, target: Location, bandwidth_gbps: f64) -> f64 {
        if source == target {
            return 0.0;
        }
        if bandwidth_gbps <= 0.0 {
            return f64::INFINITY;
        }

        // size_bytes / (bandwidth_gbps * 1e9) * 1000 = ms
        let transfer_ms = (self.size_bytes as f64 / (bandwidth_gbps * 1e9)) * 1000.0;

        // Add latency
        let latency_ms = (source.typical_latency_us() + target.typical_latency_us()) / 1000.0;

        transfer_ms + latency_ms
    }

    /// Find the best approximation within quality constraints.
    pub fn best_approximation(&self, max_quality_loss: f64) -> Option<&ApproximationOption> {
        self.approximations
            .iter()
            .filter(|a| a.quality_loss <= max_quality_loss)
            .max_by(|a, b| {
                a.size_reduction
                    .partial_cmp(&b.size_reduction)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl PartialEq for ExecutionState {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ExecutionState {}

impl std::hash::Hash for ExecutionState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_state() {
        let state = ExecutionState::new(StateType::KvCache, 1024 * 1024);
        assert_eq!(state.state_type, StateType::KvCache);
        assert_eq!(state.size_bytes, 1024 * 1024);
        assert_eq!(state.location, Location::NotMaterialized);
    }

    #[test]
    fn test_kv_cache_factory() {
        let state = ExecutionState::kv_cache(5, 1024, 32, 128, DType::Float16);
        assert_eq!(state.state_type, StateType::KvCache);
        assert_eq!(state.metadata.layer_id, Some(5));
        assert!(state.recomputable);
        // Size: 2 * 1024 * 32 * 128 * 2 = 16MB
        assert_eq!(state.size_bytes, 2 * 1024 * 32 * 128 * 2);
    }

    #[test]
    fn test_expert_weight_factory() {
        let state = ExecutionState::expert_weight(3, 7, 50 * 1024 * 1024);
        assert_eq!(state.state_type, StateType::ExpertWeight);
        assert_eq!(state.metadata.layer_id, Some(3));
        assert_eq!(state.metadata.expert_id, Some(7));
        assert!(!state.recomputable);
    }

    #[test]
    fn test_location_properties() {
        let mut state = ExecutionState::new(StateType::Activation, 1024);

        assert!(!state.is_materialized());
        assert!(!state.is_in_gpu());

        state.location = Location::Hbm;
        assert!(state.is_materialized());
        assert!(state.is_in_gpu());

        state.location = Location::Dram;
        assert!(state.is_materialized());
        assert!(!state.is_in_gpu());
    }

    #[test]
    fn test_transfer_time() {
        let state = ExecutionState::new(StateType::Activation, 1024 * 1024 * 1024); // 1GB

        // 1GB at 32 GB/s = ~31.25ms
        let time = state.transfer_time_ms(Location::Dram, Location::Hbm, 32.0);
        assert!((time - 31.25).abs() < 1.0);
    }

    #[test]
    fn test_builder_methods() {
        let state = ExecutionState::new(StateType::Activation, 1024)
            .with_location(Location::Hbm)
            .with_pinned(true)
            .with_compute_cost(5.0)
            .with_priority(Priority::High);

        assert_eq!(state.location, Location::Hbm);
        assert!(state.pinned);
        assert_eq!(state.compute_cost_ms, Some(5.0));
        assert_eq!(state.priority, Priority::High);
    }
}
