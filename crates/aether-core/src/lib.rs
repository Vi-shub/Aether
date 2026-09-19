//! # Aether Core
//!
//! Core abstractions for the Aether AI Execution Runtime.
//!
//! Aether is a state-aware AI execution runtime that jointly optimizes
//! compute and state movement. Unlike traditional inference systems that
//! treat memory management as a separate cache subsystem, Aether makes
//! state decisions (keep, move, recompute, approximate) based on the
//! execution graph.
//!
//! ## Key Concepts
//!
//! - **ExecutionState**: Every piece of data (KV cache, activations, weights)
//! - **StateDecision**: What to do with state (KEEP, MOVE, RECOMPUTE, etc.)
//! - **ExecutionGraph**: DAG of operations with state dependencies
//! - **CostModel**: Estimates cost of different actions
//! - **DecisionEngine**: Makes optimal decisions based on execution graph
//!
//! ## Example
//!
//! ```rust,ignore
//! use aether_core::{
//!     ExecutionState, StateType, Location, Action,
//!     HardwareConfig, CostModel, DecisionEngine,
//! };
//!
//! // Configure hardware
//! let hardware = HardwareConfig::builder()
//!     .hbm_capacity_gb(24)
//!     .dram_capacity_gb(64)
//!     .pcie_bandwidth_gbps(32)
//!     .build();
//!
//! // Create decision engine
//! let cost_model = CostModel::new(&hardware);
//! let engine = DecisionEngine::new(cost_model);
//!
//! // Make decisions
//! let plan = engine.plan(&states, &graph, &constraints);
//! ```

pub mod types;
pub mod state;
pub mod decision;
pub mod graph;
pub mod hardware;
pub mod cost_model;
pub mod decision_engine;
pub mod state_manager;
pub mod error;

// Re-exports
pub use types::*;
pub use state::ExecutionState;
pub use decision::{StateDecision, ExecutionPlan, DecisionStatus};
pub use graph::{ExecutionGraph, Operation};
pub use hardware::{HardwareConfig, MemoryTier};
pub use cost_model::CostModel;
pub use decision_engine::{DecisionEngine, Constraints};
pub use state_manager::StateManager;
pub use error::{AetherError, Result};

/// Aether version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
