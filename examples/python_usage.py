#!/usr/bin/env python3
"""
Aether Python API Usage Example

This example demonstrates how to use the Aether runtime to manage
AI inference state with intelligent compute vs. movement decisions.
"""

# NOTE: To use this example, first build the Python bindings:
# cd crates/aether-python
# maturin develop --release

def main():
    """Main example demonstrating Aether Python API."""

    # Import aether (after building with maturin)
    # import aether

    # For demonstration without the actual module:
    print("=" * 60)
    print("Aether Python API Example")
    print("=" * 60)

    # Example 1: Create a runtime
    print("\n1. Creating Runtime")
    print("-" * 40)
    print("""
    # Create an H100-optimized runtime
    runtime = aether.Runtime.h100()

    # Or create memory-constrained runtime (8GB GPU)
    runtime = aether.Runtime.memory_constrained(8)

    # Or create with custom config
    config = aether.RuntimeConfig(
        hardware=aether.HardwareConfig.a100(),
        enable_prefetch=True,
        enable_recompute=True,
        memory_pressure_threshold=0.85
    )
    runtime = aether.Runtime(config)
    """)

    # Example 2: Register states
    print("\n2. Registering States")
    print("-" * 40)
    print("""
    # Create KV cache state for layer 0
    kv_cache = aether.State.kv_cache(
        layer=0,
        size_bytes=128 * 1024 * 1024,  # 128 MB
        shape=[1, 2048, 32, 128],
        batch_size=1,
        seq_len=2048
    )

    # Create activation state
    activation = aether.State(
        id=1,
        state_type=aether.StateType.activation(layer=0),
        name="layer_0_activation",
        size_bytes=64 * 1024 * 1024,
        shape=[1, 2048, 4096],
        location=aether.Location.hbm(),
        dtype=aether.DType.float16(),
        recomputable=True
    )

    # Create expert weight state (for MoE)
    expert = aether.State.expert_weight(
        layer=0,
        expert=3,
        size_bytes=512 * 1024 * 1024,
        shape=[14336, 4096]
    )

    # Register with runtime
    runtime.register_state(kv_cache)
    runtime.register_state(activation)
    runtime.register_state(expert)
    """)

    # Example 3: Planning and execution
    print("\n3. Planning and Execution")
    print("-" * 40)
    print("""
    # Build execution graph for inference
    runtime.build_transformer_graph(
        num_layers=32,
        hidden_dim=4096,
        batch_size=1,
        seq_len=2048,
        is_prefill=True,
        is_moe=False
    )

    # Plan execution (Aether decides: KEEP, MOVE, RECOMPUTE, etc.)
    plan = runtime.plan()

    print(f"Plan summary: {plan.summary()}")
    print(f"Action breakdown: {plan.count_actions()}")

    # Iterate through decisions
    for decision in plan:
        print(f"  State {decision.state_id}: {decision.action} -> {decision.target_location}")

    # Execute the plan
    runtime.execute(plan)
    """)

    # Example 4: Cost analysis
    print("\n4. Cost Analysis")
    print("-" * 40)
    print("""
    # Estimate move cost
    move_cost = runtime.estimate_move_cost(
        state=kv_cache,
        from_location="cpu",
        to_location="gpu"
    )
    print(f"Move cost: {move_cost:.2f} ms")

    # Compare move vs recompute
    comparison = runtime.compare_move_vs_recompute(
        state=activation,
        from_location="nvme",
        to_location="gpu"
    )
    print(f"Move time: {comparison['move_time_ms']:.2f} ms")
    print(f"Recompute time: {comparison['recompute_time_ms']:.2f} ms")
    print(f"Should recompute: {comparison['should_recompute']}")
    print(f"Savings: {comparison['savings_ms']:.2f} ms")
    """)

    # Example 5: Metrics
    print("\n5. Monitoring Metrics")
    print("-" * 40)
    print("""
    # Get runtime metrics
    metrics = runtime.metrics()

    print(f"GPU Memory: {metrics.gpu_memory_used / 1e9:.2f} GB used")
    print(f"GPU Utilization: {metrics.gpu_memory_utilization * 100:.1f}%")
    print(f"States on GPU: {metrics.states_on_gpu}")
    print(f"States on CPU: {metrics.states_on_cpu}")
    print(f"Cache hit rate: {metrics.cache_hit_rate * 100:.1f}%")
    """)

    # Example 6: Utility functions
    print("\n6. Utility Functions")
    print("-" * 40)
    print("""
    # Estimate KV cache size
    kv_size = aether.estimate_kv_cache_size(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        batch_size=16,
        seq_len=4096,
        dtype="float16"
    )
    print(f"KV Cache size: {kv_size / 1e9:.2f} GB")

    # Estimate model size
    model_size = aether.estimate_model_size(
        num_layers=32,
        hidden_dim=4096,
        intermediate_dim=11008,
        vocab_size=32000,
        num_experts=0,  # Dense model
        dtype="float16"
    )
    print(f"Model size: {model_size / 1e9:.2f} GB")

    # MoE model
    moe_size = aether.estimate_model_size(
        num_layers=32,
        hidden_dim=4096,
        intermediate_dim=14336,
        vocab_size=32000,
        num_experts=8,  # 8 experts
        dtype="float16"
    )
    print(f"MoE model size: {moe_size / 1e9:.2f} GB")

    # Get device info
    num_devices = aether.get_device_count()
    device_info = aether.get_device_info(0)
    print(f"Devices: {num_devices}")
    print(f"Device 0: {device_info['name']}, {device_info['memory_gb']} GB")
    """)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

    # Actual calculations for demonstration
    print("\n[Demo calculations without aether module]")

    # KV cache size calculation
    kv_size = 2 * 32 * 16 * 4096 * 8 * 128 * 2
    print(f"KV Cache (32 layers, batch=16, seq=4096): {kv_size / 1e9:.2f} GB")

    # Model size calculation (dense)
    embedding = 32000 * 4096 * 2
    per_layer = 4 * 4096 * 4096 * 2 + 3 * 4096 * 11008 * 2
    model_size = embedding + 32 * per_layer
    print(f"LLaMA-7B size: {model_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
