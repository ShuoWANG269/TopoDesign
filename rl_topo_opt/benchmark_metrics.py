"""
Benchmark metrics calculation time for different topology sizes.
"""
import sys
import os
import time

# Add ATOP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ATOP'))

from generator.network import construct_topology
from NSGAII.solution import NetTopology
from converter import convert_atop_to_simplified
from env.reward import RewardCalculator

def benchmark_metrics(n_gpus, depth, width, n_runs=3):
    """Benchmark metrics calculation for given topology size."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: GPUs={n_gpus}, Depth={depth}, Width={width}")
    print(f"{'='*60}")

    # Generate topology
    print("Generating topology...")
    gen_start = time.time()

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    topology, connection_blocks, blueprint = construct_topology(
        total_gpus=n_gpus,
        total_layers=depth,
        d_max=width,
        generator=full_generator
    )
    net_topo = NetTopology(topology, connection_blocks, blueprint)
    gen_time = time.time() - gen_start

    # Convert to simplified
    simplified = convert_atop_to_simplified(net_topo)
    print(f"  Nodes: {len(simplified.nodes)}")
    print(f"  Edges: {len(simplified.edges)}")
    print(f"  Generation time: {gen_time:.2f}s")

    # Benchmark metrics calculation
    calculator = RewardCalculator()
    times = []

    for run in range(n_runs):
        start = time.time()
        cost, latency, ft = calculator.compute_metrics(simplified)
        elapsed = time.time() - start
        times.append(elapsed)

        if run == 0:
            print(f"\nMetrics (first run):")
            print(f"  Cost: {cost:.2f}")
            print(f"  Latency: {latency:.4f}")
            print(f"  Fault Tolerance: {ft:.4f}")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nMetrics calculation time ({n_runs} runs):")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Min: {min_time:.4f}s")
    print(f"  Max: {max_time:.4f}s")

    return {
        'n_gpus': n_gpus,
        'nodes': len(simplified.nodes),
        'edges': len(simplified.edges),
        'gen_time': gen_time,
        'avg_metrics_time': avg_time,
        'min_metrics_time': min_time,
        'max_metrics_time': max_time
    }

if __name__ == '__main__':
    results = []

    # Test different sizes
    configs = [
        (64, 3, 3),   # Current training size
        (128, 3, 3),  # 2x GPUs
        (256, 3, 3),  # 4x GPUs
    ]

    for n_gpus, depth, width in configs:
        result = benchmark_metrics(n_gpus, depth, width, n_runs=3)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'GPUs':<8} {'Nodes':<8} {'Edges':<8} {'Gen(s)':<10} {'Metrics(s)':<12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['n_gpus']:<8} {r['nodes']:<8} {r['edges']:<8} "
              f"{r['gen_time']:<10.2f} {r['avg_metrics_time']:<12.4f}")
