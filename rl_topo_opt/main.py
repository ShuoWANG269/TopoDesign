"""
Main entry point for topology optimization training.
"""

import argparse
import sys
import os

# Add ATOP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ATOP'))

from generator.network import construct_topology
from NSGAII.solution import NetTopology
from training.trainer import Trainer


def generate_initial_topology(n_gpus: int = 16, depth: int = 2, width: int = 3):
    """
    Generate initial topology using ATOP.

    Args:
        n_gpus: Number of GPUs
        depth: Topology depth (total_layers)
        width: Topology width (d_max)

    Returns:
        NetTopology object
    """
    print(f"Generating initial topology: GPUs={n_gpus}, Depth={depth}, Width={width}")

    # Use ATOP to generate topology
    # construct_topology returns generator(Topology, connection_blocks, blueprint)
    # Default generator is lambda x,y,z: x, so we need a custom one
    def full_generator(topology, connection_blocks, blueprint):
        return topology, connection_blocks, blueprint

    topology, connection_blocks, blueprint = construct_topology(
        total_gpus=n_gpus,
        total_layers=depth,
        d_max=width,
        generator=full_generator
    )

    # Create NetTopology object
    net_topo = NetTopology(topology, connection_blocks, blueprint)

    print(f"Generated topology with {len(topology.layers)} layers")
    for i, layer in enumerate(topology.layers):
        print(f"  Layer {i}: {len(layer)} nodes")

    return net_topo




def main():
    parser = argparse.ArgumentParser(description='Train RL agent for topology optimization')

    # Topology parameters
    parser.add_argument('--n_gpus', type=int, default=16, help='Number of GPUs')
    parser.add_argument('--depth', type=int, default=2, help='Topology depth')
    parser.add_argument('--width', type=int, default=3, help='Topology width')

    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total training timesteps')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='Checkpoint interval')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Epochs per update')
    parser.add_argument('--tensorboard_log', type=str, default='./logs', help='Tensorboard log directory')

    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Evaluation
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, do not train')
    parser.add_argument('--n_eval_episodes', type=int, default=10, help='Number of evaluation episodes')

    args = parser.parse_args()

    # Generate initial topology
    initial_net_topo = generate_initial_topology(
        n_gpus=args.n_gpus,
        depth=args.depth,
        width=args.width
    )

    # Create trainer
    trainer = Trainer(
        initial_net_topo=initial_net_topo,
        total_timesteps=args.total_timesteps,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        tensorboard_log=args.tensorboard_log
    )

    if args.eval_only:
        # Evaluation only
        if args.resume:
            checkpoint = trainer.checkpoint_manager.load_checkpoint(args.resume, trainer.model)
            print(f"Loaded checkpoint from step {checkpoint['step']}")
        else:
            print("Warning: No checkpoint specified for evaluation, using untrained model")

        trainer.evaluate(n_episodes=args.n_eval_episodes)
    else:
        # Training
        trainer.train(resume_from=args.resume)

        # Evaluate after training
        print("\nEvaluating trained model...")
        trainer.evaluate(n_episodes=args.n_eval_episodes)


if __name__ == '__main__':
    main()
