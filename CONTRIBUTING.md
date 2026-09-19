# Contributing to Aether

Thank you for your interest in contributing to Aether!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/aether-runtime/aether
cd aether
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/ -v
```

## Code Style

- We use `black` for formatting
- We use `ruff` for linting
- We use `mypy` for type checking

Run all checks:
```bash
black aether/ tests/
ruff check aether/ tests/
mypy aether/
```

## Project Structure

```
aether/
├── core/           # Core abstractions (types, state, decision, graph)
├── planner/        # Cost model and decision engine
├── state/          # State management
├── hardware/       # Hardware configuration
├── engine/         # Main runtime
├── adapters/       # Integration with vLLM, SGLang, etc.
└── utils/          # Utilities (logging, profiling)
```

## Key Concepts

### ExecutionState
The fundamental unit of state. Every piece of data (KV cache, activations, weights) is an ExecutionState.

### StateDecision
What to do with a state: KEEP, MOVE, PREFETCH, RECOMPUTE, APPROXIMATE, EVICT.

### ExecutionGraph
DAG of operations with state dependencies. Used for lookahead planning.

### CostModel
Estimates cost of different actions based on hardware and state properties.

### DecisionEngine
Uses cost model + execution graph to make optimal decisions.

## Contributing Areas

1. **Cost Model Improvements**: Better bandwidth/latency estimation
2. **Decision Engine**: More sophisticated planning algorithms
3. **Adapters**: Integration with vLLM, SGLang, transformers
4. **Benchmarks**: Real workload benchmarks
5. **Hardware Support**: CXL, NVMe GDS, multi-GPU

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a PR with a clear description

## Questions?

Open an issue on GitHub!
