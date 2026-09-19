# Aether

**State-Aware AI Execution Runtime**

> *Where compute meets state.*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

---

## The Problem

Current AI inference systems have **two separate control planes**:

```
EXECUTION PLANE              MEMORY PLANE
"What to compute next"       "Where is my data"
        │                           │
        └───── LOOSE COUPLING ──────┘
              (Memory just reacts)
```

When memory says "data not ready," execution **stalls**. The memory system doesn't know what computation is coming—it just evicts based on LRU or simple priority.

**This is fundamentally wrong.**

Memory decisions ARE execution decisions.

---

## The Aether Thesis

> **Can an inference runtime jointly plan computation and state movement rather than treating memory management as a subordinate cache subsystem?**

For any piece of state (KV cache, activations, expert weights, intermediates), Aether asks:

> "Given what computation is coming, should I **keep** it, **move** it, **recompute** it, **approximate** it, or **evict** it?"

This is **not caching**. This is **execution planning**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         AETHER                              │
│              State-Aware AI Execution Runtime               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐    ┌─────────────────┐               │
│   │ ExecutionGraph  │    │  CostModel      │               │
│   │                 │    │                 │               │
│   │ • Operations    │    │ • Move cost     │               │
│   │ • Dependencies  │    │ • Recompute cost│               │
│   │ • Future needs  │    │ • Bandwidth     │               │
│   └────────┬────────┘    └────────┬────────┘               │
│            │                      │                         │
│            └──────────┬───────────┘                         │
│                       ▼                                     │
│            ┌─────────────────────┐                          │
│            │  DecisionEngine     │                          │
│            │                     │                          │
│            │  For each state:    │                          │
│            │  ├─ KEEP            │                          │
│            │  ├─ MOVE            │                          │
│            │  ├─ PREFETCH        │                          │
│            │  ├─ RECOMPUTE  ◄────┼── Key Innovation         │
│            │  ├─ APPROXIMATE     │                          │
│            │  └─ EVICT           │                          │
│            └──────────┬──────────┘                          │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                          │
│            │   ExecutionPlan     │                          │
│            │                     │                          │
│            │  Ordered decisions  │                          │
│            │  with dependencies  │                          │
│            └──────────┬──────────┘                          │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                          │
│            │   StateManager      │                          │
│            │                     │                          │
│            │  • Execute moves    │                          │
│            │  • Track locations  │                          │
│            │  • Memory budgets   │                          │
│            └─────────────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Innovation: Move vs Recompute

Traditional systems: **State not in GPU? Load it.**

Aether: **Is loading faster than recomputing?**

```
Tensor X (100MB) in DRAM
├── MOVE to HBM:     3.1ms (at 32 GB/s)
└── RECOMPUTE:       2.0ms (single matmul)

Decision: RECOMPUTE (saves 1.1ms)
```

For NVMe storage, the difference is even larger:

```
Tensor X (500MB) on NVMe
├── LOAD to HBM:    74.9ms (at 7 GB/s)
└── RECOMPUTE:      15.0ms (MLP forward)

Decision: RECOMPUTE (5x speedup!)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/aether-runtime/aether
cd aether

# Build with Cargo
cargo build --release

# Run tests
cargo test
```

---

## Quick Start (Rust)

```rust
use aether_core::{
    ExecutionState, StateType, Location, Action,
    HardwareConfig, CostModel, DecisionEngine, Constraints,
    ExecutionGraph,
};
use std::collections::HashMap;

fn main() {
    // Configure hardware
    let hardware = HardwareConfig::builder()
        .device_name("RTX 4090")
        .hbm_capacity_gb(24.0)
        .pcie_bandwidth_gbps(32.0)
        .build();

    // Create decision engine
    let cost_model = CostModel::new(hardware.clone());
    let mut engine = DecisionEngine::new(cost_model);

    // Create some states
    let mut states = HashMap::new();
    
    let kv_cache = ExecutionState::kv_cache(0, 2048, 32, 128, DType::Float16)
        .with_location(Location::Dram);
    states.insert(kv_cache.id.clone(), kv_cache);

    // Create execution graph
    let graph = ExecutionGraph::new();

    // Plan with constraints
    let constraints = Constraints::new()
        .with_hbm_budget(20 * 1024 * 1024 * 1024) // 20GB
        .with_prefetch_horizon(8);

    let plan = engine.plan(&states, &graph, &constraints, 0);

    // Execute decisions
    for decision in &plan {
        println!("{}: {:?}", decision.state_id, decision.action);
    }
}
```

---

## Project Structure

```
aether/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── aether-core/        # Core library
│   │   └── src/
│   │       ├── types.rs        # StateType, Location, Action
│   │       ├── state.rs        # ExecutionState
│   │       ├── decision.rs     # StateDecision, ExecutionPlan
│   │       ├── graph.rs        # ExecutionGraph
│   │       ├── hardware.rs     # HardwareConfig
│   │       ├── cost_model.rs   # Cost estimation
│   │       ├── decision_engine.rs  # Core planning
│   │       └── state_manager.rs    # State tracking
│   ├── aether-cuda/        # CUDA bindings (planned)
│   └── aether-py/          # Python bindings (planned)
├── examples/               # Example code
├── benches/               # Benchmarks
└── tests/                 # Integration tests
```

---

## Core Concepts

### ExecutionState
Every piece of data in inference: KV cache, activations, weights, experts.

```rust
let state = ExecutionState::kv_cache(layer_id, seq_len, num_heads, head_dim, dtype)
    .with_location(Location::Dram)
    .with_compute_cost(5.0);  // ms to recompute
```

### Action
What to do with state:
- `Keep` - Maintain in current location
- `Move` - Transfer to different tier
- `Prefetch` - Proactive movement
- `Recompute` - Discard and recompute when needed
- `Approximate` - Use lower-fidelity version
- `Evict` - Remove from memory

### CostModel
Estimates action costs based on hardware:

```rust
let move_cost = cost_model.move_cost(&state, Location::Hbm);
let (action, cost, rationale) = cost_model.compare_move_vs_recompute(&state, &graph, &registry);
```

### DecisionEngine
Makes optimal decisions based on execution graph:

```rust
let plan = engine.plan(&states, &graph, &constraints, current_op_idx);
```

---

## Benchmarks (Target)

| Workload | Baseline | Aether | Improvement |
|----------|----------|--------|-------------|
| Mixtral 8x7B (24GB GPU) | OOM | Runs | ∞ |
| 128K context (constrained) | 45 tok/s | 60 tok/s | +33% |
| MoE expert loading | 10ms stall | 2ms overlap | -80% |

---

## Roadmap

- [x] Core abstractions (Rust)
- [x] Cost model
- [x] Decision engine
- [x] State manager
- [ ] CUDA memory operations
- [ ] Async transfer engine
- [ ] vLLM integration
- [ ] Python bindings (PyO3)
- [ ] Benchmark suite

---

## Why Rust?

| Factor | C++ | Rust | Python |
|--------|-----|------|--------|
| Performance | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Memory Safety | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| Concurrency | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| CUDA Support | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

Rust provides:
- Memory safety without GC (critical for a runtime)
- Zero-cost abstractions
- Excellent concurrency primitives
- Growing ML ecosystem (candle, burn, mistral.rs)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0

---

## Citation

```bibtex
@software{aether2026,
  title = {Aether: State-Aware AI Execution Runtime},
  year = {2026},
  url = {https://github.com/aether-runtime/aether}
}
```
