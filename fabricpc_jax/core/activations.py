"""
Activation functions for predictive coding networks in JAX.

All functions are pure and compatible with JAX transformations (jit, vmap, grad).
"""

from typing import Callable, Tuple, Dict, Any
import jax.numpy as jnp
from jax import nn


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))"""
    return nn.sigmoid(x)


def sigmoid_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))"""
    s = nn.sigmoid(x)
    return s * (1 - s)


def relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU activation: max(0, x)"""
    return nn.relu(x)


def relu_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(jnp.float32)


def tanh(x: jnp.ndarray) -> jnp.ndarray:
    """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return jnp.tanh(x)


def tanh_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of tanh: 1 - tanh²(x)"""
    t = jnp.tanh(x)
    return 1 - t**2


def linear(x: jnp.ndarray) -> jnp.ndarray:
    """Linear activation: identity function"""
    return x


def linear_deriv(x: jnp.ndarray) -> jnp.ndarray:
    """Derivative of linear: always 1"""
    return jnp.ones_like(x)


def leaky_relu(x: jnp.ndarray, alpha: float = 0.01) -> jnp.ndarray:
    """Leaky ReLU: max(alpha * x, x)"""
    return jnp.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x: jnp.ndarray, alpha: float = 0.01) -> jnp.ndarray:
    """Derivative of Leaky ReLU"""
    return jnp.where(x > 0, 1.0, alpha)


# Activation function registry
ACTIVATIONS: Dict[str, Tuple[Callable, Callable]] = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "relu": (relu, relu_deriv),
    "tanh": (tanh, tanh_deriv),
    "linear": (linear, linear_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
}


def get_activation(config: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Get activation function and its derivative from config.

    Args:
        config: Dictionary with at least {"type": "sigmoid"} or similar

    Returns:
        Tuple of (activation_fn, derivative_fn)

    Example:
        >>> config = {"type": "sigmoid"}
        >>> act_fn, deriv_fn = get_activation(config)
        >>> x = jnp.array([0.0, 1.0, -1.0])
        >>> act_fn(x)
        DeviceArray([0.5, 0.731, 0.269], dtype=float32)
    """
    act_type = config.get("type", "sigmoid").lower()

    if act_type not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation type: '{act_type}'. "
            f"Available: {list(ACTIVATIONS.keys())}"
        )

    return ACTIVATIONS[act_type]


def get_activation_fn(config: Dict[str, Any]) -> Callable:
    """Get just the activation function (without derivative)."""
    return get_activation(config)[0]


def get_activation_deriv(config: Dict[str, Any]) -> Callable:
    """Get just the activation derivative function."""
    return get_activation(config)[1]
