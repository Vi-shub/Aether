//! Decision Engine: The core brain of Aether.
//!
//! The decision engine takes execution graph, current state, and constraints,
//! and produces an ExecutionPlan with optimal state decisions.

use std::collections::{HashMap, HashSet};

use crate::cost_model::CostModel;
use crate::decision::{ExecutionPlan, StateDecision};
use crate::graph::ExecutionGraph;
use crate::hardware::HardwareConfig;
use crate::state::ExecutionState;
use crate::types::*;

// =============================================================================
// Constraints
// =============================================================================

/// Constraints for the decision engine.
#[derive(Debug, Clone)]
pub struct Constraints {
    /// Maximum latency for state readiness (ms).
    pub max_latency_ms: Option<f64>,
    /// Maximum HBM usage (bytes).
    pub hbm_budget_bytes: Option<u64>,
    /// Maximum DRAM usage (bytes).
    pub dram_budget_bytes: Option<u64>,
    /// Minimum throughput requirement (tokens/sec).
    pub min_throughput_tokens_per_sec: Option<f64>,
    /// Maximum acceptable quality degradation (0-1).
    pub max_quality_loss: f64,
    /// How many operations ahead to consider.
    pub prefetch_horizon: usize,
    /// Whether recomputation is allowed.
    pub allow_recompute: bool,
    /// Whether approximation is allowed.
    pub allow_approximation: bool,
    /// Prioritize latency over throughput.
    pub prioritize_latency: bool,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            max_latency_ms: None,
            hbm_budget_bytes: None,
            dram_budget_bytes: None,
            min_throughput_tokens_per_sec: None,
            max_quality_loss: 0.1,
            prefetch_horizon: 8,
            allow_recompute: true,
            allow_approximation: true,
            prioritize_latency: true,
        }
    }
}

impl Constraints {
    /// Create new constraints with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set HBM budget.
    pub fn with_hbm_budget(mut self, bytes: u64) -> Self {
        self.hbm_budget_bytes = Some(bytes);
        self
    }

    /// Set prefetch horizon.
    pub fn with_prefetch_horizon(mut self, horizon: usize) -> Self {
        self.prefetch_horizon = horizon;
        self
    }

    /// Set max latency.
    pub fn with_max_latency(mut self, ms: f64) -> Self {
        self.max_latency_ms = Some(ms);
        self
    }

    /// Disable recomputation.
    pub fn without_recompute(mut self) -> Self {
        self.allow_recompute = false;
        self
    }
}

// =============================================================================
// Planning Context
// =============================================================================

/// Context maintained during planning.
#[derive(Debug)]
struct PlanningContext {
    current_op_idx: usize,
    hbm_used: u64,
    dram_used: u64,
    decided_states: HashSet<StateId>,
}

// =============================================================================
// Decision Engine
// =============================================================================

/// The core decision engine for Aether.
///
/// Implements Aether's key innovation: jointly optimizing compute and
/// state movement based on the execution graph.
pub struct DecisionEngine {
    cost_model: CostModel,
    hardware: HardwareConfig,
    stats: DecisionStats,
}

/// Statistics from the decision engine.
#[derive(Debug, Default, Clone)]
pub struct DecisionStats {
    pub plans_generated: u64,
    pub decisions_made: u64,
    pub recompute_chosen: u64,
    pub move_chosen: u64,
    pub prefetch_chosen: u64,
}

impl DecisionEngine {
    /// Create a new decision engine.
    pub fn new(cost_model: CostModel) -> Self {
        let hardware = cost_model.hardware().clone();
        Self {
            cost_model,
            hardware,
            stats: DecisionStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &DecisionStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DecisionStats::default();
    }

    // =========================================================================
    // Main Planning API
    // =========================================================================

    /// Generate an execution plan for state management.
    pub fn plan(
        &mut self,
        states: &HashMap<StateId, ExecutionState>,
        graph: &ExecutionGraph,
        constraints: &Constraints,
        current_op_idx: usize,
    ) -> ExecutionPlan {
        let mut plan = ExecutionPlan::new();
        plan.horizon_ms = constraints.prefetch_horizon as f64 * 10.0;

        // Initialize context
        let mut ctx = PlanningContext {
            current_op_idx,
            hbm_used: states
                .values()
                .filter(|s| s.location == Location::Hbm)
                .map(|s| s.size_bytes)
                .sum(),
            dram_used: states
                .values()
                .filter(|s| s.location == Location::Dram)
                .map(|s| s.size_bytes)
                .sum(),
            decided_states: HashSet::new(),
        };

        // Phase 1: Identify states needed in future
        let future_demands = graph.get_future_state_demands(current_op_idx, constraints.prefetch_horizon);

        // Phase 2: Make decisions for demanded states (sorted by urgency)
        let mut sorted_demands: Vec<_> = future_demands.into_iter().collect();
        sorted_demands.sort_by(|a, b| {
            b.1 .1
                .partial_cmp(&a.1 .1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (state_id, (ops_until_needed, urgency)) in sorted_demands {
            if let Some(state) = states.get(&state_id) {
                if ctx.decided_states.contains(&state_id) {
                    continue;
                }

                if let Some(decision) = self.decide_for_state(
                    state,
                    ops_until_needed,
                    urgency,
                    graph,
                    states,
                    constraints,
                    &mut ctx,
                ) {
                    plan.add(decision);
                    ctx.decided_states.insert(state_id);
                }
            }
        }

        // Phase 3: Handle memory pressure (evictions if needed)
        if let Some(budget) = constraints.hbm_budget_bytes {
            let evictions = self.plan_evictions(states, graph, constraints, &ctx, budget);
            plan.add_all(evictions);
        }

        // Phase 4: Optimize ordering
        plan.optimize_ordering();

        self.stats.plans_generated += 1;
        self.stats.decisions_made += plan.len() as u64;

        plan
    }

    // =========================================================================
    // Decision Logic
    // =========================================================================

    fn decide_for_state(
        &mut self,
        state: &ExecutionState,
        ops_until_needed: usize,
        urgency: f64,
        graph: &ExecutionGraph,
        states: &HashMap<StateId, ExecutionState>,
        constraints: &Constraints,
        ctx: &mut PlanningContext,
    ) -> Option<StateDecision> {
        // Already in HBM and pinned? Keep it
        if state.location == Location::Hbm && state.pinned {
            return Some(
                StateDecision::keep(state.id.clone())
                    .with_rationale("Pinned in HBM"),
            );
        }

        // Already in HBM with low pressure? Keep it
        if state.location == Location::Hbm {
            let hbm_capacity = self.hardware.total_capacity(Location::Hbm);
            let hbm_pressure = ctx.hbm_used as f64 / hbm_capacity as f64;

            if hbm_pressure < 0.9 || urgency > 0.5 {
                return Some(
                    StateDecision::keep(state.id.clone())
                        .with_rationale(format!("In HBM, urgency={:.2}", urgency)),
                );
            }
        }

        // Not in HBM - decide how to get it there
        self.decide_materialization(state, ops_until_needed, graph, states, constraints, ctx)
    }

    fn decide_materialization(
        &mut self,
        state: &ExecutionState,
        ops_until_needed: usize,
        graph: &ExecutionGraph,
        states: &HashMap<StateId, ExecutionState>,
        constraints: &Constraints,
        _ctx: &mut PlanningContext,
    ) -> Option<StateDecision> {
        // Calculate deadline
        let deadline_ms = if ops_until_needed > 0 {
            let future_ops = graph.project_future(_ctx.current_op_idx, ops_until_needed);
            Some(future_ops.iter().map(|op| op.compute_cost_ms).sum())
        } else {
            constraints.max_latency_ms
        };

        // Get best action from cost model
        let (action, cost) = self.cost_model.best_action(state, graph, states, deadline_ms);

        // Check constraints
        if !constraints.allow_recompute && action == Action::Recompute {
            // Fall back to move
            let move_cost = self.cost_model.move_cost(state, Location::Hbm);
            if move_cost.is_possible() {
                self.stats.move_chosen += 1;
                return Some(
                    StateDecision::move_to(state.id.clone(), Location::Hbm, move_cost.time_ms)
                        .with_rationale("Recompute disabled, using move"),
                );
            }
        }

        // Build decision based on action
        let is_prefetch = ops_until_needed > 1;
        let decision = match action {
            Action::Keep => {
                StateDecision::keep(state.id.clone())
                    .with_rationale("Best option is to keep")
            }
            Action::Move | Action::Prefetch => {
                if is_prefetch {
                    self.stats.prefetch_chosen += 1;
                } else {
                    self.stats.move_chosen += 1;
                }
                if is_prefetch {
                    StateDecision::prefetch(state.id.clone(), Location::Hbm, cost.time_ms)
                } else {
                    StateDecision::move_to(state.id.clone(), Location::Hbm, cost.time_ms)
                }
                .with_rationale(format!(
                    "{} to HBM ({:.1}ms)",
                    if is_prefetch { "Prefetch" } else { "Move" },
                    cost.time_ms
                ))
            }
            Action::Recompute => {
                self.stats.recompute_chosen += 1;
                StateDecision::recompute(state.id.clone(), cost.time_ms)
                    .with_rationale(format!("Recompute ({:.1}ms)", cost.time_ms))
            }
            Action::Evict => {
                StateDecision::evict(state.id.clone(), Some(Location::Dram))
                    .with_rationale("Evict to free memory")
            }
            _ => {
                return None;
            }
        };

        Some(decision.with_confidence(cost.confidence))
    }

    fn plan_evictions(
        &self,
        states: &HashMap<StateId, ExecutionState>,
        _graph: &ExecutionGraph,
        _constraints: &Constraints,
        ctx: &PlanningContext,
        budget: u64,
    ) -> Vec<StateDecision> {
        let mut decisions = Vec::new();

        if ctx.hbm_used <= budget {
            return decisions;
        }

        // Find eviction candidates
        let mut candidates: Vec<_> = states
            .values()
            .filter(|s| s.location == Location::Hbm && !s.pinned && s.priority != Priority::Critical)
            .map(|s| {
                // Score: higher = evict first
                let score = (s.priority as u8 as f64) * 10.0
                    + (1.0 - s.metadata.access_probability) * 5.0
                    + s.size_mb();
                (score, s)
            })
            .collect();

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut current_usage = ctx.hbm_used;
        for (score, state) in candidates {
            if current_usage <= budget {
                break;
            }

            decisions.push(
                StateDecision::evict(state.id.clone(), Some(Location::Dram))
                    .with_rationale(format!("Evict to meet budget (score={:.1})", score))
                    .with_memory_delta(-(state.size_bytes as i64)),
            );

            current_usage = current_usage.saturating_sub(state.size_bytes);
        }

        decisions
    }

    // =========================================================================
    // Expert Scheduling (for MoE)
    // =========================================================================

    /// Plan expert placement for MoE models.
    pub fn plan_expert_placement(
        &mut self,
        experts: &HashMap<StateId, ExecutionState>,
        predicted_experts: &[(StateId, f64)], // (expert_id, probability)
        constraints: &Constraints,
    ) -> ExecutionPlan {
        let mut plan = ExecutionPlan::new();

        let hbm_budget = constraints
            .hbm_budget_bytes
            .unwrap_or(self.hardware.total_capacity(Location::Hbm));

        let mut hbm_used = 0u64;

        // Sort by prediction probability
        let mut sorted: Vec<_> = predicted_experts.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (expert_id, prob) in sorted {
            if let Some(expert) = experts.get(&expert_id) {
                if hbm_used + expert.size_bytes <= hbm_budget {
                    // Should be in HBM
                    if expert.location != Location::Hbm {
                        let cost = self.cost_model.move_cost(expert, Location::Hbm);
                        plan.add(
                            StateDecision::prefetch(expert_id.clone(), Location::Hbm, cost.time_ms)
                                .with_confidence(prob)
                                .with_rationale(format!("Prefetch expert (prob={:.2})", prob)),
                        );
                        self.stats.prefetch_chosen += 1;
                    }
                    hbm_used += expert.size_bytes;
                } else {
                    // Should be evicted
                    if expert.location == Location::Hbm {
                        plan.add(
                            StateDecision::evict(expert_id.clone(), Some(Location::Dram))
                                .with_confidence(1.0 - prob)
                                .with_rationale(format!("Evict expert (prob={:.2})", prob)),
                        );
                    }
                }
            }
        }

        plan
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
    fn test_plan_empty() {
        let cost_model = CostModel::new(test_hardware());
        let mut engine = DecisionEngine::new(cost_model);
        let graph = ExecutionGraph::new();
        let states = HashMap::new();
        let constraints = Constraints::default();

        let plan = engine.plan(&states, &graph, &constraints, 0);

        assert!(plan.is_empty());
    }

    #[test]
    fn test_constraints_builder() {
        let c = Constraints::new()
            .with_hbm_budget(10 * 1024 * 1024 * 1024)
            .with_prefetch_horizon(16)
            .without_recompute();

        assert_eq!(c.hbm_budget_bytes, Some(10 * 1024 * 1024 * 1024));
        assert_eq!(c.prefetch_horizon, 16);
        assert!(!c.allow_recompute);
    }

    #[test]
    fn test_expert_placement() {
        let cost_model = CostModel::new(test_hardware());
        let mut engine = DecisionEngine::new(cost_model);

        let mut experts = HashMap::new();
        for i in 0..8 {
            let expert = ExecutionState::expert_weight(0, i, 500 * 1024 * 1024)
                .with_location(Location::Dram);
            experts.insert(expert.id.clone(), expert);
        }

        let expert_ids: Vec<_> = experts.keys().cloned().collect();
        let predictions: Vec<_> = expert_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), 1.0 - (i as f64 * 0.1)))
            .collect();

        let constraints = Constraints::new()
            .with_hbm_budget(2 * 1024 * 1024 * 1024); // 2GB

        let plan = engine.plan_expert_placement(&experts, &predictions, &constraints);

        // Should have prefetch decisions for top experts
        let prefetch_count = plan.by_action(Action::Prefetch).count();
        assert!(prefetch_count > 0);
    }
}
