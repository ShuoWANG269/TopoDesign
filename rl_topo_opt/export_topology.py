"""
Export initial and trained topology to result directory.

Usage:
    python export_topology.py [--checkpoint PATH] [--result_dir DIR]
"""

import os
import sys
import pickle
import glob
import numpy as np

# Add parent directory to path and ATOP
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ATOP'))

from generator.network import construct_topology
from NSGAII.solution import NetTopology
from converter import convert_atop_to_simplified, SimplifiedTopology


def export_topology(
    checkpoint_path: str = None,
    result_dir: str = './result',
    n_gpus: int = 32,
    depth: int = 3,
    width: int = 2
):
    """
    Export initial and trained topology to result directory.

    Args:
        checkpoint_path: Path to checkpoint
        result_dir: Output directory
        n_gpus: Number of GPUs
        depth: Network depth
        width: Switch width
    """
    os.makedirs(result_dir, exist_ok=True)

    # Find checkpoint if not specified
    if checkpoint_path is None:
        checkpoints = glob.glob(os.path.join('./checkpoints', 'checkpoint_step_*.pkl'))
        if not checkpoints:
            raise FileNotFoundError("No checkpoint found")
        checkpoint_path = max(checkpoints, key=lambda x: int(x.split('_step_')[1].split('_')[0]))
        print(f"Using latest checkpoint: {checkpoint_path}")

    # Load checkpoint
    print("Loading checkpoint...")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    step = checkpoint['step']
    timestamp = checkpoint['timestamp']
    initial_edges = checkpoint.get('initial_edges', None)
    topology_params = checkpoint.get('topology_params', {})

    print(f"Checkpoint step: {step}")
    print(f"Timestamp: {timestamp}")

    # Use topology params from checkpoint if available
    if topology_params:
        n_gpus = topology_params.get('n_gpus', n_gpus)
        depth = topology_params.get('depth', depth)
        width = topology_params.get('width', width)
        print(f"Training topology: GPUs={n_gpus}, Depth={depth}, Width={width}")

    # Step 1: Generate topology
    print(f"\nGenerating topology: GPUs={n_gpus}, Depth={depth}, Width={width}")

    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    topology, connection_blocks, blueprint = construct_topology(
        total_gpus=n_gpus,
        total_layers=depth,
        d_max=width,
        generator=full_generator
    )
    initial_net_topo = NetTopology(topology, connection_blocks, blueprint)

    # Convert to simplified topology
    initial_simplified = convert_atop_to_simplified(initial_net_topo)
    n_edges = len(initial_simplified.edges)

    print(f"Generated topology: {len(initial_simplified.nodes)} nodes, {n_edges} edges")

    # If checkpoint has topology data, use it for a fair comparison
    if initial_edges and len(initial_edges) > 0:
        print(f"\nNote: Using training topology from checkpoint ({len(initial_edges)} edges)")
        # Recreate topology from saved edges
        all_nodes = set()
        for src, dst in initial_edges:
            all_nodes.add(src)
            all_nodes.add(dst)

        # Infer node types from original topology
        node_types = {}
        if hasattr(initial_net_topo, 'topology') and hasattr(initial_net_topo.topology, 'layers'):
            # GPU nodes are in first layer
            gpu_nodes = set(initial_net_topo.topology.layers[0]) if initial_net_topo.topology.layers else set()
            for node in all_nodes:
                node_types[node] = 'GPU' if node in gpu_nodes else 'SWITCH'
        else:
            # Fallback: assume no GPU nodes
            for node in all_nodes:
                node_types[node] = 'SWITCH'

        # Create simplified topology from saved edges
        initial_simplified = SimplifiedTopology(
            nodes=sorted(all_nodes),
            edges=initial_edges,
            node_types=node_types,
            edge_bandwidths={},
            original_net_topo=initial_net_topo
        )
        n_edges = len(initial_edges)
        print(f"Restored topology: {len(initial_simplified.nodes)} nodes, {n_edges} edges")

    # Step 2: Load trained model
    print("\nLoading trained model...")

    from env.topo_env import TopoEnv
    from models.ppo_agent import create_ppo_agent

    # Create environment with SimplifiedTopology directly
    env = TopoEnv(initial_simplified, max_steps=100)
    model = create_ppo_agent(env)

    # Load state dict - filter out action_net (different action space)
    state_dict = {k: v for k, v in checkpoint['model_state'].items()
                  if not k.startswith('action_net')}
    model.policy.load_state_dict(state_dict, strict=False)
    print(f"Model loaded (action space: {n_edges})")

    # Step 3: Run inference
    print("\nRunning inference with RL agent...")
    from env.reward import RewardCalculator

    reward_calculator = RewardCalculator()

    obs = env.reset()
    done = False
    removed_edges = []
    max_inference_steps = 100

    # Debug: check initial action mask
    initial_mask = obs['action_mask']
    valid_actions = np.sum(initial_mask)
    print(f"Initial valid actions: {valid_actions}/{len(initial_mask)}")

    step_count = 0
    while not done and step_count < max_inference_steps:
        action, _ = model.predict(obs, deterministic=True)
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)
        obs, reward, done, info = env.step(action)
        if action not in env.removed_edges:
            removed_edges.append(action)
        step_count += 1
        if step_count % 20 == 0:
            remaining = len(env.current_topo.edges)
            valid_actions = np.sum(obs['action_mask'])
            print(f"  Step {step_count}: edges_remaining={remaining}, valid_actions={valid_actions}")

    final_simplified = env.current_topo
    final_net_topo = final_simplified.original_net_topo

    print(f"\nFinal topology: {len(final_simplified.nodes)} nodes, {len(final_simplified.edges)} edges")
    print(f"Removed edges: {len(removed_edges)}")

    # Step 4: Compute metrics
    print("\nComputing metrics...")
    # Get individual metrics
    cost, latency, ft = reward_calculator.compute_metrics(initial_net_topo)
    initial_metrics = {
        'cost': cost,
        'latency': latency,
        'fault_tolerance': ft,
        'total': cost + latency - ft  # Simplified total score
    }

    cost, latency, ft = reward_calculator.compute_metrics(final_net_topo)
    final_metrics = {
        'cost': cost,
        'latency': latency,
        'fault_tolerance': ft,
        'total': cost + latency - ft
    }

    # Step 5: Save results
    result_data = {
        'parameters': {
            'checkpoint_step': step,
            'timestamp': timestamp,
            'n_gpus': n_gpus,
            'depth': depth,
            'width': width
        },
        'initial_topology': {
            'nodes': list(initial_simplified.nodes),
            'edges': list(initial_simplified.edges),
            'metrics': initial_metrics
        },
        'final_topology': {
            'nodes': list(final_simplified.nodes),
            'edges': list(final_simplified.edges),
            'removed_edges': [int(a) for a in removed_edges],
            'metrics': final_metrics
        }
    }

    # Save as pickle
    result_pkl = os.path.join(result_dir, 'topology_comparison.pkl')
    with open(result_pkl, 'wb') as f:
        pickle.dump(result_data, f)
    print(f"\nSaved: {result_pkl}")

    # Save as readable text
    result_txt = os.path.join(result_dir, 'topology_report.txt')
    with open(result_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("TOPOLOGY OPTIMIZATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Training Steps: {step}\n")
        f.write(f"  Timestamp: {timestamp}\n")
        f.write(f"  GPUs: {n_gpus}\n")
        f.write(f"  Depth: {depth}\n")
        f.write(f"  Width: {width}\n\n")

        f.write("INITIAL TOPOLOGY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Nodes: {len(initial_simplified.nodes)}\n")
        f.write(f"  Edges: {len(initial_simplified.edges)}\n")
        f.write(f"  Metrics:\n")
        f.write(f"    - Cost: {initial_metrics['cost']:.4f}\n")
        f.write(f"    - Latency: {initial_metrics['latency']:.4f}\n")
        f.write(f"    - Fault Tolerance: {initial_metrics['fault_tolerance']:.4f}\n")
        f.write(f"    - Total: {initial_metrics['total']:.4f}\n\n")

        f.write("FINAL TOPOLOGY (After RL Optimization)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Nodes: {len(final_simplified.nodes)}\n")
        f.write(f"  Edges: {len(final_simplified.edges)}\n")
        f.write(f"  Removed Edges: {len(removed_edges)}\n")
        f.write(f"  Edge Reduction: {len(removed_edges)/len(initial_simplified.edges)*100:.1f}%\n")
        f.write(f"  Metrics:\n")
        f.write(f"    - Cost: {final_metrics['cost']:.4f}\n")
        f.write(f"    - Latency: {final_metrics['latency']:.4f}\n")
        f.write(f"    - Fault Tolerance: {final_metrics['fault_tolerance']:.4f}\n")
        f.write(f"    - Total: {final_metrics['total']:.4f}\n\n")

        f.write("IMPROVEMENT\n")
        f.write("-" * 40 + "\n")
        cost_change = (final_metrics['cost'] - initial_metrics['cost']) / initial_metrics['cost'] * 100 if initial_metrics['cost'] != 0 else 0
        latency_change = (final_metrics['latency'] - initial_metrics['latency']) / initial_metrics['latency'] * 100 if initial_metrics['latency'] != 0 else 0
        ft_change = (final_metrics['fault_tolerance'] - initial_metrics['fault_tolerance']) / initial_metrics['fault_tolerance'] * 100 if initial_metrics['fault_tolerance'] != 0 else 0
        total_change = (final_metrics['total'] - initial_metrics['total']) / initial_metrics['total'] * 100 if initial_metrics['total'] != 0 else 0

        f.write(f"  Cost: {cost_change:+.1f}%\n")
        f.write(f"  Latency: {latency_change:+.1f}%\n")
        f.write(f"  Fault Tolerance: {ft_change:+.1f}%\n")
        f.write(f"  Total Score: {total_change:+.1f}%\n\n")

        f.write("EDGE LIST (Final Topology)\n")
        f.write("-" * 40 + "\n")
        for i, (src, dst) in enumerate(final_simplified.edges):
            f.write(f"  {i+1}: {src} -> {dst}\n")
        f.write("\n")

        f.write("REMOVED EDGES\n")
        f.write("-" * 40 + "\n")
        for i, action_idx in enumerate(removed_edges):
            edge = initial_simplified.edges[action_idx]
            f.write(f"  {i+1}: {edge[0]} -> {edge[1]}\n")

    print(f"Saved: {result_txt}")

    # Save edge lists as CSV
    import csv
    edges_csv = os.path.join(result_dir, 'final_edges.csv')
    with open(edges_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst'])
        for src, dst in final_simplified.edges:
            writer.writerow([src, dst])
    print(f"Saved: {edges_csv}")

    print(f"\nResults exported to: {result_dir}")

    return result_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export topology results')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--result_dir', type=str, default='./result', help='Output directory')
    parser.add_argument('--n_gpus', type=int, default=32, help='Number of GPUs')
    parser.add_argument('--depth', type=int, default=3, help='Topology depth')
    parser.add_argument('--width', type=int, default=2, help='Topology width')

    args = parser.parse_args()

    result = export_topology(
        checkpoint_path=args.checkpoint,
        result_dir=args.result_dir,
        n_gpus=args.n_gpus,
        depth=args.depth,
        width=args.width
    )
