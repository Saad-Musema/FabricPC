"""
FabricPC-JAX: Predictive Coding Networks in JAX
================================================

A functional, high-performance implementation of predictive coding networks
using JAX for automatic differentiation, JIT compilation, and multi-device parallelism.

Key Features:
- Functional programming paradigm (immutable data structures)
- JIT-compiled inference and training loops
- Multi-GPU/TPU support with pmap
- XLA optimization for maximum performance

Example:
    >>> from fabricpc_jax.models import create_pc_graph
    >>> from fabricpc_jax.training import train_pcn
    >>>
    >>> params, structure = create_pc_graph(config)
    >>> params = train_pcn(params, structure, train_loader, config)
"""

__version__ = "0.1.0"

from fabricpc_jax.core import types, activations, inference
from fabricpc_jax.models import graph_net
from fabricpc_jax.training import train_loop, optimizers

__all__ = ["types", "activations", "inference", "graph_net", "train_loop", "optimizers"]
