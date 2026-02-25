"""P2 integration tests for main.py topology rewrite wiring."""

import importlib
import sys

from add01_full_mesh_topology.full_mesh import rewrite_topology as real_rewrite_topology


class _FakeTrainer:
    """Lightweight trainer stub for CLI wiring tests."""

    last_initial_topo = None

    def __init__(self, initial_net_topo, **kwargs):
        _FakeTrainer.last_initial_topo = initial_net_topo

    def train(self, resume_from=None):
        return None

    def evaluate(self, n_episodes=10):
        return {}


def _count_undirected_edges(net_topo):
    edges = set()
    for node_id, node in net_topo.topology.nodes.items():
        for sibling in node.siblings:
            sibling_id = sibling.node_id if hasattr(sibling, "node_id") else sibling
            if sibling_id != node_id:
                edges.add(tuple(sorted((node_id, sibling_id))))
    return len(edges)


def test_main_default_rewrite_mode_none(monkeypatch):
    main_module = importlib.import_module("main")

    captured = {}

    def fake_rewrite_topology(net_topo, mode="none", default_bandwidth=1.0):
        captured["mode"] = mode
        captured["default_bandwidth"] = default_bandwidth
        return net_topo

    monkeypatch.setattr(main_module, "rewrite_topology", fake_rewrite_topology)
    monkeypatch.setattr(main_module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--n_gpus", "8",
            "--depth", "2",
            "--width", "2",
            "--eval_only",
        ],
    )

    main_module.main()

    assert captured["mode"] == "none"
    assert captured["default_bandwidth"] == 1.0


def test_main_full_mesh_rewrite_reaches_trainer(monkeypatch):
    main_module = importlib.import_module("main")

    captured = {}

    def fake_rewrite_topology(net_topo, mode="none", default_bandwidth=1.0):
        captured["mode"] = mode
        captured["default_bandwidth"] = default_bandwidth
        return real_rewrite_topology(
            net_topo,
            mode=mode,
            default_bandwidth=default_bandwidth,
        )

    monkeypatch.setattr(main_module, "rewrite_topology", fake_rewrite_topology)
    monkeypatch.setattr(main_module, "Trainer", _FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--n_gpus", "8",
            "--depth", "2",
            "--width", "2",
            "--topology_rewrite_mode", "full_mesh",
            "--full_mesh_bandwidth", "1.7",
            "--eval_only",
        ],
    )

    main_module.main()

    assert captured["mode"] == "full_mesh"
    assert captured["default_bandwidth"] == 1.7

    topo = _FakeTrainer.last_initial_topo
    n_nodes = len(topo.topology.nodes)
    assert _count_undirected_edges(topo) == n_nodes * (n_nodes - 1) // 2
