//! Decision engine benchmarks.

use std::time::Instant;
use serde::{Deserialize, Serialize};
use tracing::info;

use aether_core::{
    CostModel, DecisionEngine, ExecutionGraph, ExecutionState, HardwareConfig,
    Location, StateType, DType, Priority, TransformerGraphBuilder,
};

use crate::scenarios::DecisionScenario;

/// Decision benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResult {
    /// Number of states.
    pub num_states: usize,
    /// Number of operations.
    pub num_operations: usize,
    /// Memory pressure.
    pub memory_pressure: f64,
    /// Planning time (ms).
    pub planning_time_ms: f64,
    /// Number of decisions made.
    pub num_decisions: usize,
    /// Decisions per millisecond.
    pub decisions_per_ms: f64,
}

impl DecisionResult {
    /// Format as table row.
    pub fn to_row(&self) -> Vec<String> {
        vec![
            format!("{}", self.num_states),
            format!("{}", self.num_operations),
            format!("{:.0}%", self.memory_pressure * 100.0),
            format!("{:.3}", self.planning_time_ms),
            format!("{}", self.num_decisions),
            format!("{:.0}", self.decisions_per_ms),
        ]
    }
}

/// Decision engine benchmark.
pub struct DecisionBenchmark {
    scenario: DecisionScenario,
    hardware_config: HardwareConfig,
}

impl DecisionBenchmark {
    /// Create a new decision benchmark.
    pub fn new(scenario: DecisionScenario) -> Self {
        Self {
            scenario,
            hardware_config: HardwareConfig::h100(),
        }
    }

    /// Run all benchmarks.
    pub fn run(&self) -> Vec<DecisionResult> {
        let mut results = Vec::new();

        info!("Starting decision engine benchmarks...");

        for &num_states in &self.scenario.num_states {
            for &num_operations in &self.scenario.num_operations {
                for &memory_pressure in &self.scenario.memory_pressures {
                    info!(
                        "Benchmarking: {} states, {} ops, {:.0}% memory pressure",
                        num_states, num_operations, memory_pressure * 100.0
                    );

                    let result = self.benchmark_configuration(
                        num_states,
                        num_operations,
                        memory_pressure,
                    );
                    results.push(result);
                }
            }
        }

        results
    }

    /// Benchmark a specific configuration.
    fn benchmark_configuration(
        &self,
        num_states: usize,
        num_operations: usize,
        memory_pressure: f64,
    ) -> DecisionResult {
        // Create states
        let states = self.generate_states(num_states);

        // Create execution graph
        let graph = self.generate_graph(num_operations);

        // Create decision engine
        let cost_model = CostModel::new(self.hardware_config.clone());
        let engine = DecisionEngine::new(
            std::sync::Arc::new(cost_model),
            self.hardware_config.clone(),
        );

        // Run multiple iterations
        let mut total_time = 0.0;
        let mut total_decisions = 0;

        for _ in 0..self.scenario.iterations {
            let start = Instant::now();
            let plan = engine.plan(&states, &graph);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;

            total_time += elapsed;
            total_decisions += plan.len();
        }

        let avg_time = total_time / self.scenario.iterations as f64;
        let avg_decisions = total_decisions / self.scenario.iterations;

        DecisionResult {
            num_states,
            num_operations,
            memory_pressure,
            planning_time_ms: avg_time,
            num_decisions: avg_decisions,
            decisions_per_ms: avg_decisions as f64 / avg_time,
        }
    }

    /// Generate test states.
    fn generate_states(&self, count: usize) -> Vec<ExecutionState> {
        (0..count)
            .map(|i| {
                let state_type = match i % 4 {
                    0 => StateType::KvCache { layer: i % 32 },
                    1 => StateType::Activation { layer: i % 32 },
                    2 => StateType::Weight { layer: i % 32 },
                    _ => StateType::ExpertWeight { layer: i % 32, expert: i % 8 },
                };

                let location = match i % 3 {
                    0 => Location::Hbm,
                    1 => Location::Dram,
                    _ => Location::Nvme,
                };

                ExecutionState {
                    id: i as u64,
                    state_type,
                    name: format!("state_{}", i),
                    location,
                    size_bytes: (1 + (i % 10)) as u64 * 1024 * 1024, // 1-10 MB
                    dtype: DType::Float16,
                    shape: vec![1024, 1024],
                    produced_by: if i > 0 { Some((i - 1) as u64) } else { None },
                    consumed_by: vec![(i + 1) as u64],
                    depends_on: if i > 0 { vec![(i - 1) as u64] } else { vec![] },
                    compute_cost_ms: Some(0.1 * (i % 10) as f64),
                    priority: if i % 5 == 0 { Priority::High } else { Priority::Normal },
                    pinned: i % 20 == 0,
                    recomputable: i % 2 == 0,
                    approximations: vec![],
                    current_approximation: None,
                    metadata: Default::default(),
                }
            })
            .collect()
    }

    /// Generate test execution graph.
    fn generate_graph(&self, num_operations: usize) -> ExecutionGraph {
        let builder = TransformerGraphBuilder::new(num_operations, 4096);
        builder.build_prefill(1, 512, false)
    }
}

/// Summary of decision benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSummary {
    pub avg_planning_time_ms: f64,
    pub max_planning_time_ms: f64,
    pub min_planning_time_ms: f64,
    pub avg_decisions_per_ms: f64,
    pub total_benchmarks: usize,
}

impl DecisionSummary {
    /// Compute summary from results.
    pub fn from_results(results: &[DecisionResult]) -> Self {
        if results.is_empty() {
            return Self {
                avg_planning_time_ms: 0.0,
                max_planning_time_ms: 0.0,
                min_planning_time_ms: 0.0,
                avg_decisions_per_ms: 0.0,
                total_benchmarks: 0,
            };
        }

        let times: Vec<f64> = results.iter().map(|r| r.planning_time_ms).collect();
        let decisions_per_ms: Vec<f64> = results.iter().map(|r| r.decisions_per_ms).collect();

        Self {
            avg_planning_time_ms: times.iter().sum::<f64>() / times.len() as f64,
            max_planning_time_ms: times.iter().cloned().fold(0.0, f64::max),
            min_planning_time_ms: times.iter().cloned().fold(f64::MAX, f64::min),
            avg_decisions_per_ms: decisions_per_ms.iter().sum::<f64>() / decisions_per_ms.len() as f64,
            total_benchmarks: results.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_states() {
        let benchmark = DecisionBenchmark::new(DecisionScenario::default());
        let states = benchmark.generate_states(100);
        assert_eq!(states.len(), 100);
    }

    #[test]
    fn test_decision_summary() {
        let results = vec![
            DecisionResult {
                num_states: 100,
                num_operations: 32,
                memory_pressure: 0.5,
                planning_time_ms: 1.0,
                num_decisions: 50,
                decisions_per_ms: 50.0,
            },
            DecisionResult {
                num_states: 1000,
                num_operations: 64,
                memory_pressure: 0.75,
                planning_time_ms: 5.0,
                num_decisions: 500,
                decisions_per_ms: 100.0,
            },
        ];

        let summary = DecisionSummary::from_results(&results);
        assert_eq!(summary.total_benchmarks, 2);
        assert!(summary.avg_planning_time_ms > 0.0);
    }
}
