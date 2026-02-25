"""Type definitions for topology rewrite configuration."""

from dataclasses import dataclass
from enum import Enum


class TopologyRewriteMode(str, Enum):
    """Supported topology rewrite modes."""

    NONE = "none"
    FULL_MESH = "full_mesh"


@dataclass(frozen=True)
class TopologyRewriteConfig:
    """Configuration for optional topology rewrite."""

    mode: TopologyRewriteMode = TopologyRewriteMode.NONE
    full_mesh_bandwidth: float = 1.0
