//! State Manager: Manages all execution state in Aether.
//!
//! The state manager tracks all state objects, their locations,
//! and executes state decisions from the planning engine.

use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::decision::{ExecutionPlan, StateDecision, DecisionStatus};
use crate::error::{AetherError, Result};
use crate::hardware::HardwareConfig;
use crate::state::ExecutionState;
use crate::types::*;

// =============================================================================
// Statistics
// =============================================================================

/// Statistics from the state manager.
#[derive(Debug, Default)]
pub struct ManagerStats {
    pub states_registered: AtomicU64,
    pub states_evicted: AtomicU64,
    pub transfers_completed: AtomicU64,
    pub bytes_transferred: AtomicU64,
    pub recomputes_performed: AtomicU64,
}

impl ManagerStats {
    fn increment(&self, field: &AtomicU64) {
        field.fetch_add(1, Ordering::Relaxed);
    }

    fn add_bytes(&self, bytes: u64) {
        self.bytes_transferred.fetch_add(bytes, Ordering::Relaxed);
    }
}

// =============================================================================
// State Manager
// =============================================================================

/// Manages all execution state in Aether.
///
/// Thread-safe registry for all state objects with:
/// - Fast lookup by ID
/// - Indexing by location and type
/// - Memory tracking per tier
/// - Decision execution
pub struct StateManager {
    /// Hardware configuration.
    hardware: Arc<HardwareConfig>,

    /// State registry (thread-safe).
    states: DashMap<StateId, ExecutionState>,

    /// Index by location.
    by_location: DashMap<Location, HashSet<StateId>>,

    /// Index by type.
    by_type: DashMap<StateType, HashSet<StateId>>,

    /// Memory usage per tier.
    tier_usage: DashMap<Location, u64>,

    /// Statistics.
    pub stats: ManagerStats,
}

impl StateManager {
    /// Create a new state manager.
    pub fn new(hardware: HardwareConfig) -> Self {
        Self {
            hardware: Arc::new(hardware),
            states: DashMap::new(),
            by_location: DashMap::new(),
            by_type: DashMap::new(),
            tier_usage: DashMap::new(),
            stats: ManagerStats::default(),
        }
    }

    /// Get hardware configuration.
    pub fn hardware(&self) -> &HardwareConfig {
        &self.hardware
    }

    // =========================================================================
    // State Registration
    // =========================================================================

    /// Register a new state.
    pub fn register(&self, state: ExecutionState) {
        let id = state.id.clone();
        let location = state.location;
        let state_type = state.state_type;
        let size = state.size_bytes;

        // Remove old if exists
        if self.states.contains_key(&id) {
            self.unregister(&id);
        }

        // Add to registry
        self.states.insert(id.clone(), state);

        // Update location index
        self.by_location
            .entry(location)
            .or_default()
            .insert(id.clone());

        // Update type index
        self.by_type
            .entry(state_type)
            .or_default()
            .insert(id.clone());

        // Update memory tracking
        if location.is_materialized() {
            *self.tier_usage.entry(location).or_default() += size;
        }

        self.stats.increment(&self.stats.states_registered);
    }

    /// Unregister a state.
    pub fn unregister(&self, state_id: &StateId) -> Option<ExecutionState> {
        if let Some((_, state)) = self.states.remove(state_id) {
            // Remove from location index
            if let Some(mut set) = self.by_location.get_mut(&state.location) {
                set.remove(state_id);
            }

            // Remove from type index
            if let Some(mut set) = self.by_type.get_mut(&state.state_type) {
                set.remove(state_id);
            }

            // Update memory tracking
            if state.location.is_materialized() {
                if let Some(mut usage) = self.tier_usage.get_mut(&state.location) {
                    *usage = usage.saturating_sub(state.size_bytes);
                }
            }

            Some(state)
        } else {
            None
        }
    }

    // =========================================================================
    // State Queries
    // =========================================================================

    /// Get a state by ID.
    pub fn get(&self, state_id: &StateId) -> Option<ExecutionState> {
        self.states.get(state_id).map(|r| r.value().clone())
    }

    /// Check if a state exists.
    pub fn contains(&self, state_id: &StateId) -> bool {
        self.states.contains_key(state_id)
    }

    /// Get all states as a HashMap.
    pub fn get_all(&self) -> HashMap<StateId, ExecutionState> {
        self.states
            .iter()
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect()
    }

    /// Get states by location.
    pub fn get_by_location(&self, location: Location) -> Vec<ExecutionState> {
        self.by_location
            .get(&location)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.states.get(id).map(|r| r.value().clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get states by type.
    pub fn get_by_type(&self, state_type: StateType) -> Vec<ExecutionState> {
        self.by_type
            .get(&state_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.states.get(id).map(|r| r.value().clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a state is in HBM.
    pub fn is_in_hbm(&self, state_id: &StateId) -> bool {
        self.states
            .get(state_id)
            .map(|s| s.location == Location::Hbm)
            .unwrap_or(false)
    }

    /// Number of states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    // =========================================================================
    // Memory Management
    // =========================================================================

    /// Get current memory usage at a location.
    pub fn get_usage(&self, location: Location) -> u64 {
        self.tier_usage.get(&location).map(|r| *r).unwrap_or(0)
    }

    /// Get available memory at a location.
    pub fn get_available(&self, location: Location) -> u64 {
        let total = self.hardware.total_capacity(location);
        let used = self.get_usage(location);
        total.saturating_sub(used)
    }

    /// Check if size can fit at location.
    pub fn can_fit(&self, size_bytes: u64, location: Location) -> bool {
        size_bytes <= self.get_available(location)
    }

    // =========================================================================
    // Plan Execution
    // =========================================================================

    /// Execute all decisions in a plan.
    pub fn execute_plan(&self, plan: &mut ExecutionPlan) -> Result<()> {
        for decision in plan.decisions.iter_mut() {
            if decision.status == DecisionStatus::Pending {
                self.execute_decision(decision)?;
            }
        }
        Ok(())
    }

    /// Execute a single decision.
    pub fn execute_decision(&self, decision: &mut StateDecision) -> Result<()> {
        decision.status = DecisionStatus::InProgress;

        let result = match decision.action {
            Action::Keep => Ok(()),
            Action::Move | Action::Prefetch => {
                self.execute_move(&decision.state_id, decision.target_location.unwrap_or(Location::Hbm))
            }
            Action::Recompute => self.execute_recompute(&decision.state_id),
            Action::Evict => {
                self.execute_evict(&decision.state_id, decision.target_location)
            }
            _ => Ok(()),
        };

        match result {
            Ok(()) => {
                decision.status = DecisionStatus::Completed;
                Ok(())
            }
            Err(e) => {
                decision.status = DecisionStatus::Failed;
                decision.error_message = Some(e.to_string());
                Err(e)
            }
        }
    }

    fn execute_move(&self, state_id: &StateId, target: Location) -> Result<()> {
        let mut state = self.states.get_mut(state_id)
            .ok_or_else(|| AetherError::StateNotFound(state_id.clone()))?;

        let source = state.location;
        if source == target {
            return Ok(());
        }

        // Check space at target
        if !self.can_fit(state.size_bytes, target) {
            return Err(AetherError::InsufficientMemory {
                location: target,
                required: state.size_bytes,
                available: self.get_available(target),
            });
        }

        // Update indexes
        if let Some(mut set) = self.by_location.get_mut(&source) {
            set.remove(state_id);
        }
        self.by_location
            .entry(target)
            .or_default()
            .insert(state_id.clone());

        // Update memory tracking
        if source.is_materialized() {
            if let Some(mut usage) = self.tier_usage.get_mut(&source) {
                *usage = usage.saturating_sub(state.size_bytes);
            }
        }
        if target.is_materialized() {
            *self.tier_usage.entry(target).or_default() += state.size_bytes;
        }

        // Update state location
        let bytes = state.size_bytes;
        state.location = target;

        self.stats.increment(&self.stats.transfers_completed);
        self.stats.add_bytes(bytes);

        Ok(())
    }

    fn execute_recompute(&self, state_id: &StateId) -> Result<()> {
        let mut state = self.states.get_mut(state_id)
            .ok_or_else(|| AetherError::StateNotFound(state_id.clone()))?;

        if !state.recomputable {
            return Err(AetherError::NotRecomputable(state_id.clone()));
        }

        // Check space in HBM
        if !self.can_fit(state.size_bytes, Location::Hbm) {
            return Err(AetherError::InsufficientMemory {
                location: Location::Hbm,
                required: state.size_bytes,
                available: self.get_available(Location::Hbm),
            });
        }

        let source = state.location;

        // Update indexes
        if let Some(mut set) = self.by_location.get_mut(&source) {
            set.remove(state_id);
        }
        self.by_location
            .entry(Location::Hbm)
            .or_default()
            .insert(state_id.clone());

        // Update memory tracking
        if source.is_materialized() {
            if let Some(mut usage) = self.tier_usage.get_mut(&source) {
                *usage = usage.saturating_sub(state.size_bytes);
            }
        }
        *self.tier_usage.entry(Location::Hbm).or_default() += state.size_bytes;

        // Update state
        state.location = Location::Hbm;

        self.stats.increment(&self.stats.recomputes_performed);

        Ok(())
    }

    fn execute_evict(&self, state_id: &StateId, target: Option<Location>) -> Result<()> {
        let state = self.states.get(state_id)
            .ok_or_else(|| AetherError::StateNotFound(state_id.clone()))?;

        if state.pinned {
            return Err(AetherError::StatePinned(state_id.clone()));
        }
        drop(state);

        let target = target.unwrap_or(Location::Evicted);

        if target == Location::Evicted {
            // Full eviction
            if let Some(state) = self.unregister(state_id) {
                self.stats.increment(&self.stats.states_evicted);
            }
        } else {
            // Move to lower tier
            self.execute_move(state_id, target)?;
        }

        Ok(())
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /// Clear all state.
    pub fn clear(&self) {
        self.states.clear();
        self.by_location.clear();
        self.by_type.clear();
        self.tier_usage.clear();
    }

    /// Get a summary of current state.
    pub fn summary(&self) -> ManagerSummary {
        let mut by_location = HashMap::new();
        for item in self.by_location.iter() {
            by_location.insert(*item.key(), item.value().len());
        }

        let mut by_type = HashMap::new();
        for item in self.by_type.iter() {
            by_type.insert(*item.key(), item.value().len());
        }

        let mut memory_usage = HashMap::new();
        for item in self.tier_usage.iter() {
            let total = self.hardware.total_capacity(*item.key());
            memory_usage.insert(*item.key(), (*item.value(), total));
        }

        ManagerSummary {
            total_states: self.states.len(),
            by_location,
            by_type,
            memory_usage,
        }
    }
}

/// Summary of state manager.
#[derive(Debug)]
pub struct ManagerSummary {
    pub total_states: usize,
    pub by_location: HashMap<Location, usize>,
    pub by_type: HashMap<StateType, usize>,
    pub memory_usage: HashMap<Location, (u64, u64)>, // (used, total)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> StateManager {
        StateManager::new(HardwareConfig::builder().hbm_capacity_gb(24.0).build())
    }

    #[test]
    fn test_register_state() {
        let manager = test_manager();

        let state = ExecutionState::new(StateType::KvCache, 1024 * 1024)
            .with_location(Location::Hbm);
        let id = state.id.clone();

        manager.register(state);

        assert!(manager.contains(&id));
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_unregister_state() {
        let manager = test_manager();

        let state = ExecutionState::new(StateType::KvCache, 1024 * 1024);
        let id = state.id.clone();

        manager.register(state);
        assert!(manager.contains(&id));

        manager.unregister(&id);
        assert!(!manager.contains(&id));
    }

    #[test]
    fn test_memory_tracking() {
        let manager = test_manager();

        let state = ExecutionState::new(StateType::KvCache, 1024 * 1024 * 1024) // 1GB
            .with_location(Location::Hbm);

        manager.register(state);

        assert_eq!(manager.get_usage(Location::Hbm), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_execute_move() {
        let manager = test_manager();

        let state = ExecutionState::new(StateType::KvCache, 1024 * 1024)
            .with_location(Location::Dram);
        let id = state.id.clone();

        manager.register(state);

        manager.execute_move(&id, Location::Hbm).unwrap();

        let state = manager.get(&id).unwrap();
        assert_eq!(state.location, Location::Hbm);
    }
}
