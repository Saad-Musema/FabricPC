# FabricPC

**A flexible, performant predictive coding library**

FabricPC implements predictive coding networks using a clean abstraction of:
- **Nodes**: State variables (latents), projection functions, and activations
- **Wires**: Connections (edges) between nodes in the model architecture
- **Updates**: Iterative inference and learning algorithms

Uses JAX for GPU acceleration and local (node-level) automatic differentiation.

## About Predictive Coding
Predictive coding (PC) is a biologically-inspired framework for perception and learning in the brain. It posits that the brain continuously generates predictions about sensory inputs and updates its internal representations based on local prediction errors. 
The predictive coding algorithm iteratively processes credit assignment and is equivalent to backpropagation of error under certain assumptions. It runs slower than backpropagation but has advantages of:
- Potential for faster inference on specialized hardware
- Natural handling of recurrent and arbitrary architectures
- Associative memory capabilities
- Potential novel plasticity rules for continual learning

There are various flavors of PC. FabricPC provides a graph-based implementation that focuses on principles:
- Local learning rules
- Parallel processing of nodes
- Expectation-maximization style inference.
- Modularity of components
- Arbitrary architectures
- Scalability with JAX
- Extensibility for research
 
## Quick Start
```bash
# Install in editable mode (recommended for development, and running examples)
pip install -e ".[dev,torch,viz]"

# Run an example
python examples/mnist_demo.py
```

## Features
- Modular node and wire abstractions for flexible model construction
- Inherently supports arbitrary architectures: feedforward, recurrent, skip connections, etc.
- Support for various node types: Linear, Conv1D/2D/3D (planned), Transfomers (planned)
- Local automatic differentiation for efficient inference and learning
- JAX backend for GPU acceleration and scalability

## Contributions
Contributions are welcome! Please open issues or pull requests on the GitHub repository.

This is a research-first project. APIs may change frequently until v1.0 release. Any breaking changes will be documented in the changelog.

All demos must match baseline results and test suites must pass before merging new code.

## License: private until officially released. Please do not distribute.

## Extensible Nodes
 - **Custom Nodes**: Easily create new node types by subclassing `BaseNode`
 - Single decorator to register a node class - no need to modify core code
 - Example custom node: examples/custom_node.py
 - External packages can register nodes via setuptools entry points.

**External package's `pyproject.toml`:**
```toml
[project.entry-points."fabricpc.nodes"]
myconv2d = "fabricpc_conv.nodes:MyConv2DNode"
mytransformer = "fabricpc_transformer.nodes:MyTransformerBlock"
```

## Shape Conventions

 All shapes use batch-first, channels-last format (NHWC, NLC, NDHWC):

 - Consistent with JAX's default conv behavior
 - Linear: shape=(features,) - e.g., (128,) for 128-dimensional vector
 - 1D Conv: shape=(seq_len, channels) - e.g., (100, 32) for 100 timesteps, 32 channels
 - 2D Conv: shape=(H, W, C) - e.g., (28, 28, 64) for 28x28 image, 64 channels (NHWC)
 - 3D Conv: shape=(D, H, W, C) - e.g., (32, 32, 32, 16) for 3D volume

Linear nodes flatten their inputs for transformation and then reshape their outputs to the specified shape.

Conv Node Shape Flow (Future Reference)

 - Input:  (batch, H_in, W_in, C_in)   e.g., (32, 28, 28, 1)
 - Kernel: (kH, kW, C_in, C_out)       e.g., (3, 3, 1, 64)
 - Output: (batch, H_out, W_out, C_out) e.g., (32, 26, 26, 64)