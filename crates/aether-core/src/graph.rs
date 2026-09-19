//! ExecutionGraph: DAG representation of AI inference computation.
//!
//! The execution graph captures:
//! - Operations to be performed
//! - State dependencies between operations
//! - Future computation path for planning

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::types::*;

// =============================================================================
// Operation
// =============================================================================

/// A single operation in the execution graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    /// Unique identifier.
    pub id: OperationId,
    /// Type of operation.
    pub op_type: OperationType,
    /// Human-readable name.
    pub name: String,

    /// States consumed by this operation.
    pub inputs: Vec<StateId>,
    /// States produced by this operation.
    pub outputs: Vec<StateId>,

    /// Estimated compute time in milliseconds.
    pub compute_cost_ms: f64,
    /// Peak memory usage during execution.
    pub memory_bytes: u64,
    /// Floating point operations.
    pub flops: u64,

    /// Layer this operation belongs to.
    pub layer_id: Option<LayerId>,
    /// Expert this operation belongs to (for MoE).
    pub expert_id: Option<ExpertId>,

    /// True if compute-bound, False if memory-bound.
    pub is_compute_bound: bool,
    /// Can this overlap with data movement?
    pub can_overlap: bool,
}

impl Operation {
    /// Create a new operation.
    pub fn new(id: impl Into<String>, op_type: OperationType) -> Self {
        let id_str = id.into();
        Self {
            id: OperationId(id_str.clone()),
            op_type,
            name: id_str,
            inputs: Vec::new(),
            outputs: Vec::new(),
            compute_cost_ms: 0.0,
            memory_bytes: 0,
            flops: 0,
            layer_id: None,
            expert_id: None,
            is_compute_bound: true,
            can_overlap: true,
        }
    }

    /// Add an input state.
    pub fn with_input(mut self, state: StateId) -> Self {
        self.inputs.push(state);
        self
    }

    /// Add an output state.
    pub fn with_output(mut self, state: StateId) -> Self {
        self.outputs.push(state);
        self
    }

    /// Set compute cost.
    pub fn with_compute_cost(mut self, cost_ms: f64) -> Self {
        self.compute_cost_ms = cost_ms;
        self
    }

    /// Set memory usage.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Set layer ID.
    pub fn with_layer(mut self, layer_id: LayerId) -> Self {
        self.layer_id = Some(layer_id);
        self
    }
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Operation {}

impl std::hash::Hash for Operation {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// =============================================================================
// Execution Graph
// =============================================================================

/// DAG representation of AI inference computation.
///
/// The execution graph is the key abstraction for Aether's planning.
/// It captures all operations, state dependencies, and enables
/// lookahead planning for state prefetching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionGraph {
    /// All operations indexed by ID.
    operations: HashMap<OperationId, Operation>,

    /// Maps each state to the operation that produces it.
    state_producers: HashMap<StateId, OperationId>,

    /// Maps each state to operations that consume it.
    state_consumers: HashMap<StateId, Vec<OperationId>>,

    /// Operations in execution order.
    execution_order: Vec<OperationId>,
}

impl ExecutionGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an operation to the graph.
    pub fn add_operation(&mut self, op: Operation) {
        // Track state producers
        for output in &op.outputs {
            self.state_producers.insert(output.clone(), op.id.clone());
        }

        // Track state consumers
        for input in &op.inputs {
            self.state_consumers
                .entry(input.clone())
                .or_default()
                .push(op.id.clone());
        }

        // Add to execution order
        self.execution_order.push(op.id.clone());

        // Store operation
        self.operations.insert(op.id.clone(), op);
    }

    /// Get an operation by ID.
    pub fn get_operation(&self, id: &OperationId) -> Option<&Operation> {
        self.operations.get(id)
    }

    /// Get the operation that produces a state.
    pub fn get_producer(&self, state_id: &StateId) -> Option<&Operation> {
        self.state_producers
            .get(state_id)
            .and_then(|op_id| self.operations.get(op_id))
    }

    /// Get all operations that consume a state.
    pub fn get_consumers(&self, state_id: &StateId) -> Vec<&Operation> {
        self.state_consumers
            .get(state_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.operations.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get operations that must complete before the given operation.
    pub fn get_dependencies(&self, op_id: &OperationId) -> Vec<&Operation> {
        self.operations
            .get(op_id)
            .map(|op| {
                op.inputs
                    .iter()
                    .filter_map(|state_id| self.get_producer(state_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    // =========================================================================
    // Future Projection
    // =========================================================================

    /// Project the next N operations from current position.
    pub fn project_future(&self, current_idx: usize, horizon: usize) -> Vec<&Operation> {
        let end = (current_idx + horizon).min(self.execution_order.len());
        self.execution_order[current_idx..end]
            .iter()
            .filter_map(|id| self.operations.get(id))
            .collect()
    }

    /// Get states that will be needed in upcoming operations.
    ///
    /// Returns a map of state_id -> (ops_until_needed, urgency_score).
    pub fn get_future_state_demands(
        &self,
        current_idx: usize,
        horizon: usize,
    ) -> HashMap<StateId, (usize, f64)> {
        let future_ops = self.project_future(current_idx, horizon);
        let mut demands = HashMap::new();

        for (i, op) in future_ops.iter().enumerate() {
            for input in &op.inputs {
                if !demands.contains_key(input) {
                    // Urgency decreases with distance
                    let urgency = 1.0 / (i as f64 + 1.0);
                    demands.insert(input.clone(), (i, urgency));
                }
            }
        }

        demands
    }

    /// Get when a state will be needed in future execution.
    pub fn get_state_demand_curve(&self, state_id: &StateId, current_idx: usize) -> Vec<(usize, OperationId)> {
        self.execution_order[current_idx..]
            .iter()
            .enumerate()
            .filter_map(|(i, op_id)| {
                self.operations.get(op_id).and_then(|op| {
                    if op.inputs.contains(state_id) {
                        Some((i, op_id.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Find what's needed to recompute a state.
    pub fn find_recompute_info(&self, state_id: &StateId) -> Option<(Vec<StateId>, f64)> {
        self.get_producer(state_id)
            .map(|producer| (producer.inputs.clone(), producer.compute_cost_ms))
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Total number of operations.
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Total number of unique states.
    pub fn num_states(&self) -> usize {
        self.state_producers.len()
    }

    /// Total compute time (sequential).
    pub fn total_compute_ms(&self) -> f64 {
        self.operations.values().map(|op| op.compute_cost_ms).sum()
    }

    /// Iterate over operations in execution order.
    pub fn iter(&self) -> impl Iterator<Item = &Operation> {
        self.execution_order
            .iter()
            .filter_map(|id| self.operations.get(id))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_operation() {
        let mut graph = ExecutionGraph::new();

        let op = Operation::new("op1", OperationType::Attention)
            .with_input("input1".into())
            .with_output("output1".into())
            .with_compute_cost(1.0);

        graph.add_operation(op);

        assert_eq!(graph.num_operations(), 1);
        assert!(graph.state_producers.contains_key(&"output1".into()));
    }

    #[test]
    fn test_get_producer_consumer() {
        let mut graph = ExecutionGraph::new();

        let op1 = Operation::new("op1", OperationType::Linear)
            .with_output("state1".into());

        let op2 = Operation::new("op2", OperationType::Attention)
            .with_input("state1".into())
            .with_output("state2".into());

        graph.add_operation(op1);
        graph.add_operation(op2);

        let producer = graph.get_producer(&"state1".into());
        assert!(producer.is_some());
        assert_eq!(producer.unwrap().id.0, "op1");

        let consumers = graph.get_consumers(&"state1".into());
        assert_eq!(consumers.len(), 1);
        assert_eq!(consumers[0].id.0, "op2");
    }

    #[test]
    fn test_future_projection() {
        let mut graph = ExecutionGraph::new();

        for i in 0..10 {
            graph.add_operation(Operation::new(format!("op{}", i), OperationType::Linear));
        }

        let future = graph.project_future(3, 5);
        assert_eq!(future.len(), 5);
        assert_eq!(future[0].id.0, "op3");
        assert_eq!(future[4].id.0, "op7");
    }

    #[test]
    fn test_state_demands() {
        let mut graph = ExecutionGraph::new();

        let op1 = Operation::new("op1", OperationType::Linear)
            .with_input("weight".into())
            .with_output("act1".into());

        let op2 = Operation::new("op2", OperationType::Attention)
            .with_input("act1".into())
            .with_input("kv".into())
            .with_output("act2".into());

        graph.add_operation(op1);
        graph.add_operation(op2);

        let demands = graph.get_future_state_demands(0, 10);
        
        assert!(demands.contains_key(&"weight".into()));
        assert!(demands.contains_key(&"kv".into()));
    }
}
