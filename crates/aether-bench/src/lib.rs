//! # Aether Benchmarking Harness
//!
//! Comprehensive benchmarking for Aether runtime components:
//! - Memory transfer performance (H2D, D2H, D2D)
//! - Decision engine throughput
//! - State manager operations
//! - End-to-end inference scenarios

pub mod scenarios;
pub mod transfer;
pub mod decision;
pub mod memory;
pub mod report;

pub use scenarios::{BenchmarkScenario, InferenceScenario, ModelConfig};
pub use transfer::{TransferBenchmark, TransferResult};
pub use decision::{DecisionBenchmark, DecisionResult};
pub use memory::{MemoryBenchmark, MemoryResult};
pub use report::{BenchmarkReport, ReportFormat};
