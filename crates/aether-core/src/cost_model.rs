//! Cost Model for Aether Decision Engine.
//!
//! The cost model estimates the cost of different actions:
//! - MOVE: Cost of transferring state between memory tiers
//! - RECOMPUTE: Cost of recomputing state from dependencies
//! - APPROXIMATE: Cost and quality impact of approximation

use std::collections::HashMap;

use crate::error::Result;
use crate::graph::ExecutionGraph;
use crate::hardware::HardwareConfig;
use crate::state::ExecutionState;
use crate::types::*;

// =============================================================================
// Action Cost
// =============================================================================

/// Cost estimate for a single action.
#[derive(Debug, Clone)]
pub struct ActionCost {
    /// The action type.
    pub action: Action,
    /// Estimated execution time in milliseconds.
    pub time_ms: f64,
    /// Change in memory usage (bytes, can be negative).
    pub memory_delta_bytes: i64,
    /// Quality degradation (0-1, for approximation).
    pub quality_loss: f64,
    /// Confidence in this estimate (0-1).
    pub confidence: f64,
    /// Detailed cost breakdown.
    pub breakdown: HashMap<String, f64>,
}

impl ActionCost {
    /// Create a new action cost.
    pub fn new(action: Action, time_ms: f64) -> Self {
        Self {
            action,
            time_ms,
            memory_delta_bytes: 0,
            quality_loss: 0.0,
            confidence: 1.0,
            breakdown: HashMap::new(),
        }
    }

    /// Combined cost metric (time + weighted quality loss).
    pub fn total_cost(&self) -> f64 {
        // Quality loss is weighted heavily
        self.time_ms + (self.quality_loss * 100.0)
    }

    /// Create an infinite cost (action not possible).
    pub fn impossible(action: Action) -> Self {
        Self {
            action,
            time_ms: f64::INFINITY,
            memory_delta_bytes: 0,
            quality_loss: 0.0,
            confidence: 0.0,
            breakdown: HashMap::new(),
        }
    }

    /// Check if action is possible.
    pub fn is_possible(&self) -> bool {
        self.time_ms.is_finite()
    }
}

impl PartialEq for ActionCost {
    fn eq(&self, other: &Self) -> bool {
        (self.total_cost() - other.total_cost()).abs() < 1e-9
    }
}

impl PartialOrd for ActionCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.total_cost().partial_cmp(&other.total_cost())
    }
}

// =============================================================================
// Recompute Info
// =============================================================================

/// Information about recomputing a state.
#[derive(Debug, Clone)]
pub struct RecomputeInfo {
    /// Whether recomputation is possible.
    pub can_recompute: bool,
    /// Time to recompute.
    pub compute_cost_ms: f64,
    /// States needed for recomputation.
    pub required_states: Vec<StateId>,
    /// Required states that are not in HBM.
    pub missing_states: Vec<StateId>,
    /// Total cost including loading missing states.
    pub total_cost_ms: f64,
}

// =============================================================================
// Cost Model
// =============================================================================

/// Cost model for Aether's decision engine.
///
/// The cost model answers: "What is the cost of action A for state S?"
///
/// It considers:
/// - Hardware capabilities (bandwidth, compute)
/// - Current system state (memory pressure)
/// - State properties (size, type, dependencies)
/// - Future computation (what will be needed when)
pub struct CostModel {
    /// Hardware configuration.
    hardware: HardwareConfig,
    /// Contention multiplier (1.0 = no contention).
    contention_factor: f64,
}

impl CostModel {
    /// Create a new cost model.
    pub fn new(hardware: HardwareConfig) -> Self {
        Self {
            hardware,
            contention_factor: 1.0,
        }
    }

    /// Set contention factor.
    pub fn with_contention(mut self, factor: f64) -> Self {
        self.contention_factor = factor;
        self
    }

    /// Get hardware reference.
    pub fn hardware(&self) -> &HardwareConfig {
        &self.hardware
    }

    // =========================================================================
    // Core Cost Functions
    // =========================================================================

    /// Calculate cost of moving state to target location.
    pub fn move_cost(&self, state: &ExecutionState, target: Location) -> ActionCost {
        let source = state.location;

        // Same location = no-op
        if source == target {
            return ActionCost::new(Action::Keep, 0.0);
        }

        // Not materialized = can't move
        if !source.is_materialized() {
            return ActionCost::impossible(Action::Move);
        }

        // Get bandwidth
        let bandwidth = self.hardware.get_bandwidth(source, target);
        if bandwidth <= 0.0 {
            return ActionCost::impossible(Action::Move);
        }

        // Calculate transfer time
        let transfer_ms = (state.size_bytes as f64 / (bandwidth * 1e9)) * 1000.0;

        // Apply contention
        let transfer_ms = transfer_ms * self.contention_factor;

        // Add latency
        let latency_ms = (source.typical_latency_us() + target.typical_latency_us()) / 1000.0;

        let total_ms = transfer_ms + latency_ms;

        // Memory delta: new allocation at target
        let memory_delta = if target.is_materialized() {
            state.size_bytes as i64
        } else {
            0
        } - if source.is_materialized() && source != target {
            state.size_bytes as i64
        } else {
            0
        };

        let mut cost = ActionCost::new(Action::Move, total_ms);
        cost.memory_delta_bytes = memory_delta;
        cost.breakdown.insert("transfer_ms".to_string(), transfer_ms);
        cost.breakdown.insert("latency_ms".to_string(), latency_ms);
        cost.breakdown.insert("bandwidth_gbps".to_string(), bandwidth);
        cost
    }

    /// Calculate cost of recomputing a state.
    pub fn recompute_cost(
        &self,
        state_id: &StateId,
        graph: &ExecutionGraph,
        state_registry: &HashMap<StateId, ExecutionState>,
    ) -> RecomputeInfo {
        // Find the operation that produces this state
        let producer = graph.get_producer(state_id);

        if producer.is_none() {
            return RecomputeInfo {
                can_recompute: false,
                compute_cost_ms: f64::INFINITY,
                required_states: Vec::new(),
                missing_states: Vec::new(),
                total_cost_ms: f64::INFINITY,
            };
        }

        let producer = producer.unwrap();
        let required_states = producer.inputs.clone();

        // Check which states are available in HBM
        let missing_states: Vec<StateId> = required_states
            .iter()
            .filter(|s| {
                state_registry
                    .get(*s)
                    .map(|state| !state.is_in_gpu())
                    .unwrap_or(true)
            })
            .cloned()
            .collect();

        // Base compute cost
        let compute_cost_ms = producer.compute_cost_ms;

        // Add cost of loading missing states
        let load_cost_ms: f64 = missing_states
            .iter()
            .filter_map(|s| state_registry.get(s))
            .map(|state| self.move_cost(state, Location::Hbm).time_ms)
            .sum();

        let total_cost_ms = compute_cost_ms + load_cost_ms;

        RecomputeInfo {
            can_recompute: true,
            compute_cost_ms,
            required_states,
            missing_states,
            total_cost_ms,
        }
    }

    /// Calculate cost of keeping state in current location.
    pub fn keep_cost(&self, state: &ExecutionState) -> ActionCost {
        let mut cost = 0.0;

        // Add opportunity cost based on memory pressure
        if state.location == Location::Hbm {
            if let Some(tier) = self.hardware.get_tier(Location::Hbm) {
                let utilization = tier.utilization();
                if utilization > 0.9 {
                    // High pressure - cost to keep increases
                    cost = (utilization - 0.9) * 10.0;
                }
            }
        }

        ActionCost::new(Action::Keep, cost)
    }

    /// Calculate cost of evicting a state.
    pub fn evict_cost(&self, state: &ExecutionState) -> ActionCost {
        // Immediate cost is 0, but future cost depends on reuse probability
        let future_cost = if state.metadata.access_probability > 0.5 {
            state.metadata.access_probability * 5.0
        } else {
            0.0
        };

        let mut cost = ActionCost::new(Action::Evict, future_cost);
        cost.memory_delta_bytes = -(state.size_bytes as i64);
        cost.breakdown
            .insert("access_probability".to_string(), state.metadata.access_probability);
        cost
    }

    // =========================================================================
    // Decision Support
    // =========================================================================

    /// Compare move vs recompute for a state.
    ///
    /// Returns (chosen_action, cost_ms, rationale).
    pub fn compare_move_vs_recompute(
        &self,
        state: &ExecutionState,
        graph: &ExecutionGraph,
        state_registry: &HashMap<StateId, ExecutionState>,
    ) -> (Action, f64, String) {
        let move_cost = self.move_cost(state, Location::Hbm);
        let recompute_info = self.recompute_cost(&state.id, graph, state_registry);

        if !recompute_info.can_recompute {
            return (
                Action::Move,
                move_cost.time_ms,
                "Cannot recompute, must move".to_string(),
            );
        }

        let move_ms = move_cost.time_ms;
        let recompute_ms = recompute_info.total_cost_ms;

        if recompute_ms < move_ms {
            let savings = move_ms - recompute_ms;
            (
                Action::Recompute,
                recompute_ms,
                format!(
                    "Recompute saves {:.1}ms vs move ({:.1}ms vs {:.1}ms)",
                    savings, recompute_ms, move_ms
                ),
            )
        } else {
            let savings = recompute_ms - move_ms;
            (
                Action::Move,
                move_ms,
                format!(
                    "Move saves {:.1}ms vs recompute ({:.1}ms vs {:.1}ms)",
                    savings, move_ms, recompute_ms
                ),
            )
        }
    }

    /// Find the best action for a state.
    pub fn best_action(
        &self,
        state: &ExecutionState,
        graph: &ExecutionGraph,
        state_registry: &HashMap<StateId, ExecutionState>,
        deadline_ms: Option<f64>,
    ) -> (Action, ActionCost) {
        let mut options: Vec<(Action, ActionCost)> = Vec::new();

        // KEEP
        options.push((Action::Keep, self.keep_cost(state)));

        // MOVE to HBM
        if state.location != Location::Hbm {
            let cost = self.move_cost(state, Location::Hbm);
            if cost.is_possible() {
                options.push((Action::Move, cost));
            }
        }

        // RECOMPUTE
        if state.recomputable {
            let info = self.recompute_cost(&state.id, graph, state_registry);
            if info.can_recompute {
                let cost = ActionCost::new(Action::Recompute, info.total_cost_ms);
                options.push((Action::Recompute, cost));
            }
        }

        // EVICT
        options.push((Action::Evict, self.evict_cost(state)));

        // Filter by deadline
        if let Some(deadline) = deadline_ms {
            options.retain(|(_, cost)| cost.time_ms <= deadline);
        }

        // Find minimum cost
        options
            .into_iter()
            .min_by(|(_, a), (_, b)| {
                a.total_cost()
                    .partial_cmp(&b.total_cost())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((Action::Delay, ActionCost::new(Action::Delay, 0.0)))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hardware() -> HardwareConfig {
        HardwareConfig::builder()
            .hbm_capacity_gb(24.0)
            .pcie_bandwidth_gbps(32.0)
            .build()
    }

    #[test]
    fn test_move_cost() {
        let cost_model = CostModel::new(test_hardware());

        let state = ExecutionState::new(StateType::KvCache, 100 * 1024 * 1024) // 100MB
            .with_location(Location::Dram);

        let cost = cost_model.move_cost(&state, Location::Hbm);

        // 100MB at 32GB/s ≈ 3.125ms
        assert!(cost.is_possible());
        assert!((cost.time_ms - 3.125).abs() < 0.5);
    }

    #[test]
    fn test_move_same_location() {
        let cost_model = CostModel::new(test_hardware());

        let state = ExecutionState::new(StateType::KvCache, 100 * 1024 * 1024)
            .with_location(Location::Hbm);

        let cost = cost_model.move_cost(&state, Location::Hbm);

        assert_eq!(cost.action, Action::Keep);
        assert_eq!(cost.time_ms, 0.0);
    }

    #[test]
    fn test_compare_move_vs_recompute() {
        let hardware = test_hardware();
        let cost_model = CostModel::new(hardware);

        // State that's fast to recompute
        let mut state = ExecutionState::new(StateType::Activation, 100 * 1024 * 1024)
            .with_location(Location::Dram)
            .with_compute_cost(2.0);

        // Create a graph with a producer
        let mut graph = ExecutionGraph::new();
        let op = crate::graph::Operation::new("producer", OperationType::Linear)
            .with_output(state.id.clone())
            .with_compute_cost(2.0);
        graph.add_operation(op);

        let registry: HashMap<StateId, ExecutionState> =
            [(state.id.clone(), state.clone())].into_iter().collect();

        let (action, _, _) = cost_model.compare_move_vs_recompute(&state, &graph, &registry);

        // Recompute (2ms) should be faster than move (~3ms)
        assert_eq!(action, Action::Recompute);
    }
}
