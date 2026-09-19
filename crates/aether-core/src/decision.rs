//! StateDecision and ExecutionPlan: The output of Aether's decision engine.
//!
//! A StateDecision represents what to do with a single piece of state.
//! An ExecutionPlan is a sequence of decisions to execute.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::*;

// =============================================================================
// Decision Status
// =============================================================================

/// Status of a decision in the execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DecisionStatus {
    /// Decision has not been executed yet.
    Pending = 0,
    /// Decision is currently being executed.
    InProgress = 1,
    /// Decision has been successfully executed.
    Completed = 2,
    /// Decision failed to execute.
    Failed = 3,
    /// Decision was cancelled.
    Cancelled = 4,
    /// Decision was skipped (e.g., state already in target location).
    Skipped = 5,
}

impl Default for DecisionStatus {
    fn default() -> Self {
        Self::Pending
    }
}

// =============================================================================
// State Decision
// =============================================================================

/// A decision about what to do with a piece of execution state.
///
/// This is the fundamental output of Aether's decision engine. For each
/// state object, the engine produces a decision that specifies:
///
/// - What action to take (keep, move, recompute, etc.)
/// - Where to move it (if applicable)
/// - When to execute the decision
/// - Why this decision was made
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDecision {
    // =========================================================================
    // Core Decision
    // =========================================================================
    /// The state this decision applies to.
    pub state_id: StateId,
    /// What action to take.
    pub action: Action,

    // =========================================================================
    // Action Parameters
    // =========================================================================
    /// Target location for MOVE/PREFETCH/REPLICATE actions.
    pub target_location: Option<Location>,
    /// Method to use for APPROXIMATE action.
    pub approximation_method: Option<ApproximationMethod>,

    // =========================================================================
    // Timing
    // =========================================================================
    /// When to execute this decision, relative to plan start (milliseconds).
    pub target_time_ms: f64,
    /// Must complete by this time (milliseconds from plan start).
    pub deadline_ms: Option<f64>,

    // =========================================================================
    // Cost/Benefit
    // =========================================================================
    /// Estimated time to execute this decision.
    pub expected_cost_ms: f64,
    /// Estimated latency savings from this decision.
    pub expected_benefit_ms: f64,
    /// Change in memory usage (positive = increase).
    pub memory_delta_bytes: i64,

    // =========================================================================
    // Confidence and Explanation
    // =========================================================================
    /// Confidence in this decision (0-1 scale).
    pub confidence: f64,
    /// Human-readable explanation for this decision.
    pub rationale: String,

    // =========================================================================
    // Execution Tracking
    // =========================================================================
    /// Current execution status.
    pub status: DecisionStatus,
    /// Actual time taken (after execution).
    pub actual_cost_ms: Option<f64>,
    /// Error message if failed.
    pub error_message: Option<String>,

    // =========================================================================
    // Dependencies
    // =========================================================================
    /// Other states whose decisions must complete first.
    pub depends_on: Vec<StateId>,
    /// Operations that are blocked until this completes.
    pub blocks: Vec<OperationId>,
}

impl StateDecision {
    /// Create a new decision.
    pub fn new(state_id: StateId, action: Action) -> Self {
        Self {
            state_id,
            action,
            target_location: None,
            approximation_method: None,
            target_time_ms: 0.0,
            deadline_ms: None,
            expected_cost_ms: 0.0,
            expected_benefit_ms: 0.0,
            memory_delta_bytes: 0,
            confidence: 1.0,
            rationale: String::new(),
            status: DecisionStatus::Pending,
            actual_cost_ms: None,
            error_message: None,
            depends_on: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// Create a KEEP decision.
    pub fn keep(state_id: StateId) -> Self {
        Self::new(state_id, Action::Keep)
    }

    /// Create a MOVE decision.
    pub fn move_to(state_id: StateId, target: Location, cost_ms: f64) -> Self {
        let mut d = Self::new(state_id, Action::Move);
        d.target_location = Some(target);
        d.expected_cost_ms = cost_ms;
        d
    }

    /// Create a PREFETCH decision.
    pub fn prefetch(state_id: StateId, target: Location, cost_ms: f64) -> Self {
        let mut d = Self::new(state_id, Action::Prefetch);
        d.target_location = Some(target);
        d.expected_cost_ms = cost_ms;
        d
    }

    /// Create a RECOMPUTE decision.
    pub fn recompute(state_id: StateId, cost_ms: f64) -> Self {
        let mut d = Self::new(state_id, Action::Recompute);
        d.expected_cost_ms = cost_ms;
        d
    }

    /// Create an EVICT decision.
    pub fn evict(state_id: StateId, target: Option<Location>) -> Self {
        let mut d = Self::new(state_id, Action::Evict);
        d.target_location = target;
        d
    }

    // =========================================================================
    // Builder Methods
    // =========================================================================

    /// Set the rationale.
    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = rationale.into();
        self
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set expected benefit.
    pub fn with_benefit(mut self, benefit_ms: f64) -> Self {
        self.expected_benefit_ms = benefit_ms;
        self
    }

    /// Set memory delta.
    pub fn with_memory_delta(mut self, delta_bytes: i64) -> Self {
        self.memory_delta_bytes = delta_bytes;
        self
    }

    /// Set deadline.
    pub fn with_deadline(mut self, deadline_ms: f64) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }

    /// Add a dependency.
    pub fn with_dependency(mut self, dep: StateId) -> Self {
        self.depends_on.push(dep);
        self
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /// Check if this is a movement action.
    #[inline]
    pub fn is_move_action(&self) -> bool {
        self.action.is_movement()
    }

    /// Check if this involves computation.
    #[inline]
    pub fn is_compute_action(&self) -> bool {
        self.action.is_compute()
    }

    /// Net benefit (benefit - cost).
    #[inline]
    pub fn net_benefit_ms(&self) -> f64 {
        self.expected_benefit_ms - self.expected_cost_ms
    }

    /// Check if decision is pending.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.status == DecisionStatus::Pending
    }

    /// Check if decision is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.status == DecisionStatus::Completed
    }

    // =========================================================================
    // Status Updates
    // =========================================================================

    /// Mark decision as in progress.
    pub fn mark_in_progress(&mut self) {
        self.status = DecisionStatus::InProgress;
    }

    /// Mark decision as completed.
    pub fn mark_completed(&mut self, actual_cost_ms: f64) {
        self.status = DecisionStatus::Completed;
        self.actual_cost_ms = Some(actual_cost_ms);
    }

    /// Mark decision as failed.
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = DecisionStatus::Failed;
        self.error_message = Some(error.into());
    }

    /// Mark decision as skipped.
    pub fn mark_skipped(&mut self, reason: impl Into<String>) {
        self.status = DecisionStatus::Skipped;
        self.rationale = reason.into();
    }
}

// =============================================================================
// Execution Plan
// =============================================================================

/// A sequence of state decisions to execute.
///
/// The ExecutionPlan is the complete output of the decision engine.
/// It contains:
///
/// - An ordered list of decisions
/// - Dependencies between decisions
/// - Timing constraints
/// - Expected overall impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// State decisions in execution order.
    pub decisions: Vec<StateDecision>,

    /// How far ahead this plan looks (milliseconds).
    pub horizon_ms: f64,

    /// Plan is stale after this duration.
    pub valid_for: Duration,

    /// Sum of all decision costs.
    pub expected_total_cost_ms: f64,

    /// Sum of all decision benefits.
    pub expected_total_benefit_ms: f64,

    /// Net memory change from this plan.
    pub expected_memory_delta_bytes: i64,

    /// Additional planning metadata.
    pub metadata: HashMap<String, String>,

    /// When this plan was created.
    #[serde(skip)]
    created_at: Option<Instant>,
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionPlan {
    /// Create a new empty plan.
    pub fn new() -> Self {
        Self {
            decisions: Vec::new(),
            horizon_ms: 100.0,
            valid_for: Duration::from_millis(100),
            expected_total_cost_ms: 0.0,
            expected_total_benefit_ms: 0.0,
            expected_memory_delta_bytes: 0,
            metadata: HashMap::new(),
            created_at: Some(Instant::now()),
        }
    }

    /// Create an empty plan.
    pub fn empty() -> Self {
        Self::new()
    }

    /// Add a decision to the plan.
    pub fn add(&mut self, decision: StateDecision) {
        self.expected_total_cost_ms += decision.expected_cost_ms;
        self.expected_total_benefit_ms += decision.expected_benefit_ms;
        self.expected_memory_delta_bytes += decision.memory_delta_bytes;
        self.decisions.push(decision);
    }

    /// Add multiple decisions.
    pub fn add_all(&mut self, decisions: impl IntoIterator<Item = StateDecision>) {
        for d in decisions {
            self.add(d);
        }
    }

    /// Get a decision by state ID.
    pub fn get(&self, state_id: &StateId) -> Option<&StateDecision> {
        self.decisions.iter().find(|d| &d.state_id == state_id)
    }

    /// Get a mutable decision by state ID.
    pub fn get_mut(&mut self, state_id: &StateId) -> Option<&mut StateDecision> {
        self.decisions.iter_mut().find(|d| &d.state_id == state_id)
    }

    /// Remove a decision by state ID.
    pub fn remove(&mut self, state_id: &StateId) -> Option<StateDecision> {
        if let Some(idx) = self.decisions.iter().position(|d| &d.state_id == state_id) {
            let removed = self.decisions.remove(idx);
            self.expected_total_cost_ms -= removed.expected_cost_ms;
            self.expected_total_benefit_ms -= removed.expected_benefit_ms;
            self.expected_memory_delta_bytes -= removed.memory_delta_bytes;
            Some(removed)
        } else {
            None
        }
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Iterate over decisions that are ready to execute.
    pub fn ready_decisions(&self) -> impl Iterator<Item = &StateDecision> {
        let completed: std::collections::HashSet<_> = self
            .decisions
            .iter()
            .filter(|d| d.status == DecisionStatus::Completed)
            .map(|d| &d.state_id)
            .collect();

        self.decisions.iter().filter(move |d| {
            d.status == DecisionStatus::Pending
                && d.depends_on.iter().all(|dep| completed.contains(dep))
        })
    }

    /// Iterate over pending decisions.
    pub fn pending_decisions(&self) -> impl Iterator<Item = &StateDecision> {
        self.decisions
            .iter()
            .filter(|d| d.status == DecisionStatus::Pending)
    }

    /// Get decisions by action type.
    pub fn by_action(&self, action: Action) -> impl Iterator<Item = &StateDecision> {
        self.decisions.iter().filter(move |d| d.action == action)
    }

    /// Get decisions sorted by target time.
    pub fn by_target_time(&self) -> Vec<&StateDecision> {
        let mut sorted: Vec<_> = self.decisions.iter().collect();
        sorted.sort_by(|a, b| {
            a.target_time_ms
                .partial_cmp(&b.target_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /// Check if plan has no decisions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Number of decisions.
    #[inline]
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Check if all decisions are completed.
    pub fn is_complete(&self) -> bool {
        self.decisions
            .iter()
            .all(|d| d.status == DecisionStatus::Completed)
    }

    /// Check if any decisions failed.
    pub fn has_failures(&self) -> bool {
        self.decisions
            .iter()
            .any(|d| d.status == DecisionStatus::Failed)
    }

    /// Net benefit of the entire plan.
    #[inline]
    pub fn net_benefit_ms(&self) -> f64 {
        self.expected_total_benefit_ms - self.expected_total_cost_ms
    }

    /// Fraction of decisions completed.
    pub fn completion_rate(&self) -> f64 {
        if self.decisions.is_empty() {
            return 1.0;
        }
        let completed = self
            .decisions
            .iter()
            .filter(|d| d.status == DecisionStatus::Completed)
            .count();
        completed as f64 / self.decisions.len() as f64
    }

    /// Check if plan has expired.
    pub fn is_stale(&self) -> bool {
        self.created_at
            .map(|t| t.elapsed() > self.valid_for)
            .unwrap_or(false)
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Count decisions by action type.
    pub fn action_counts(&self) -> HashMap<Action, usize> {
        let mut counts = HashMap::new();
        for d in &self.decisions {
            *counts.entry(d.action).or_insert(0) += 1;
        }
        counts
    }

    /// Get a summary of the plan.
    pub fn summary(&self) -> PlanSummary {
        PlanSummary {
            num_decisions: self.decisions.len(),
            action_counts: self.action_counts(),
            total_cost_ms: self.expected_total_cost_ms,
            total_benefit_ms: self.expected_total_benefit_ms,
            net_benefit_ms: self.net_benefit_ms(),
            memory_delta_mb: self.expected_memory_delta_bytes as f64 / (1024.0 * 1024.0),
            completion_rate: self.completion_rate(),
            is_complete: self.is_complete(),
            has_failures: self.has_failures(),
        }
    }

    // =========================================================================
    // Optimization
    // =========================================================================

    /// Reorder decisions to minimize total execution time.
    pub fn optimize_ordering(&mut self) {
        // Simple topological sort respecting dependencies
        let mut completed: std::collections::HashSet<StateId> = std::collections::HashSet::new();
        let mut ordered: Vec<StateDecision> = Vec::with_capacity(self.decisions.len());
        let mut remaining: Vec<StateDecision> = std::mem::take(&mut self.decisions);

        while !remaining.is_empty() {
            // Find decisions with all dependencies satisfied
            let mut ready_indices: Vec<usize> = remaining
                .iter()
                .enumerate()
                .filter(|(_, d)| d.depends_on.iter().all(|dep| completed.contains(dep)))
                .map(|(i, _)| i)
                .collect();

            if ready_indices.is_empty() {
                // Circular dependency or missing dependency - add remaining in order
                ordered.extend(remaining);
                break;
            }

            // Sort ready decisions by target_time, then by benefit
            ready_indices.sort_by(|&a, &b| {
                let da = &remaining[a];
                let db = &remaining[b];
                da.target_time_ms
                    .partial_cmp(&db.target_time_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        db.expected_benefit_ms
                            .partial_cmp(&da.expected_benefit_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });

            // Process in reverse order to maintain correct indices
            ready_indices.reverse();
            for i in ready_indices {
                let d = remaining.remove(i);
                completed.insert(d.state_id.clone());
                ordered.push(d);
            }
        }

        self.decisions = ordered;
    }

    /// Merge another plan into this one.
    pub fn merge(&mut self, other: ExecutionPlan) {
        for d in other.decisions {
            if self.get(&d.state_id).is_none() {
                self.add(d);
            }
        }
    }
}

impl IntoIterator for ExecutionPlan {
    type Item = StateDecision;
    type IntoIter = std::vec::IntoIter<StateDecision>;

    fn into_iter(self) -> Self::IntoIter {
        self.decisions.into_iter()
    }
}

impl<'a> IntoIterator for &'a ExecutionPlan {
    type Item = &'a StateDecision;
    type IntoIter = std::slice::Iter<'a, StateDecision>;

    fn into_iter(self) -> Self::IntoIter {
        self.decisions.iter()
    }
}

// =============================================================================
// Plan Summary
// =============================================================================

/// Summary statistics for an execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub num_decisions: usize,
    pub action_counts: HashMap<Action, usize>,
    pub total_cost_ms: f64,
    pub total_benefit_ms: f64,
    pub net_benefit_ms: f64,
    pub memory_delta_mb: f64,
    pub completion_rate: f64,
    pub is_complete: bool,
    pub has_failures: bool,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_decision() {
        let d = StateDecision::new("test".into(), Action::Move);
        assert_eq!(d.action, Action::Move);
        assert!(d.is_pending());
        assert!(d.is_move_action());
    }

    #[test]
    fn test_decision_factories() {
        let keep = StateDecision::keep("s1".into());
        assert_eq!(keep.action, Action::Keep);

        let move_d = StateDecision::move_to("s2".into(), Location::Hbm, 5.0);
        assert_eq!(move_d.action, Action::Move);
        assert_eq!(move_d.target_location, Some(Location::Hbm));

        let recompute = StateDecision::recompute("s3".into(), 3.0);
        assert_eq!(recompute.action, Action::Recompute);
        assert!(recompute.is_compute_action());
    }

    #[test]
    fn test_plan_add() {
        let mut plan = ExecutionPlan::new();

        let d1 = StateDecision::move_to("s1".into(), Location::Hbm, 5.0).with_benefit(10.0);
        let d2 = StateDecision::recompute("s2".into(), 3.0);

        plan.add(d1);
        plan.add(d2);

        assert_eq!(plan.len(), 2);
        assert_eq!(plan.expected_total_cost_ms, 8.0);
        assert_eq!(plan.expected_total_benefit_ms, 10.0);
    }

    #[test]
    fn test_plan_action_counts() {
        let mut plan = ExecutionPlan::new();

        plan.add(StateDecision::prefetch("s1".into(), Location::Hbm, 1.0));
        plan.add(StateDecision::prefetch("s2".into(), Location::Hbm, 1.0));
        plan.add(StateDecision::recompute("s3".into(), 1.0));

        let counts = plan.action_counts();
        assert_eq!(counts.get(&Action::Prefetch), Some(&2));
        assert_eq!(counts.get(&Action::Recompute), Some(&1));
    }

    #[test]
    fn test_plan_completion() {
        let mut plan = ExecutionPlan::new();
        plan.add(StateDecision::keep("s1".into()));
        plan.add(StateDecision::keep("s2".into()));

        assert!(!plan.is_complete());
        assert_eq!(plan.completion_rate(), 0.0);

        plan.decisions[0].mark_completed(0.0);
        assert_eq!(plan.completion_rate(), 0.5);

        plan.decisions[1].mark_completed(0.0);
        assert!(plan.is_complete());
    }
}
