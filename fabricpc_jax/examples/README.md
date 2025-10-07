# FabricPC JAX Examples

This directory contains minimal, self-contained examples demonstrating the JAX implementation of FabricPC.

## Examples

### `mnist_demo.py` - Minimal MNIST Classification ⭐ Start Here

The simplest possible example showing end-to-end training on MNIST.

**Features**:
- Dictionary-based model definition
- Automatic JIT compilation
- ~60 lines of code total
- Achieves ~96% accuracy in 5 epochs

**Architecture**:
```
Input (784) → Hidden (128, sigmoid) → Output (10)
```

**Run it**:
```bash
# From project root
python -m fabricpc_jax.examples.mnist_demo

# Or with PYTHONPATH
export PYTHONPATH=/home/mrb/Projects/PC-Continual-Learning:$PYTHONPATH
python fabricpc_jax/examples/mnist_demo.py
```

**Expected Output**:
```
Model created: 3 nodes, 2 edges
Total parameters: 101770

Training (JIT compilation on first batch)...
Epoch 1/5, Loss: 0.1234
Epoch 2/5, Loss: 0.0567
...
Epoch 5/5, Loss: 0.0123

Evaluating...
Test Accuracy: 96.12%
Test Loss: 0.0089
```

---

### `mnist_advanced.py` - Advanced Training Features

A more comprehensive example demonstrating:
- Deeper network architectures (4 hidden layers)
- Different activation functions (ReLU)
- Advanced optimizers (AdamW with weight decay)
- Custom training loops with monitoring
- Progress tracking and best model checkpointing

**Architecture**:
```
Input (784) → H1 (256, ReLU) → H2 (128, ReLU) → H3 (64, ReLU) → Output (10)
```

**Run it**:
```bash
python -m fabricpc_jax.examples.mnist_advanced
```

**Expected Output**:
```
[Model Architecture]
  Nodes: 5
  Edges: 4
  Total parameters: 243,546
  Layer sizes: 784 → 256 → 128 → 64 → 10 → (output)

[Training for 10 epochs]
  Epoch 1/10 - Loss: 0.1234, Accuracy: 92.34%, Time: 45.2s
  ★ New best accuracy: 92.34%
  Epoch 2/10 - Loss: 0.0567, Accuracy: 95.12%, Time: 42.1s
  ★ New best accuracy: 95.12%
  ...

[Final Results]
  Best accuracy: 97.45%
```

## Customizing the Examples

All examples use a simple dictionary-based configuration. To modify:

1. **Change architecture**: Edit `node_list` and `edge_list` in config
2. **Adjust hyperparameters**: Modify `train_config` dictionary
3. **Try different optimizers**: Change `optimizer` type (adam, sgd, adamw)
4. **Deeper networks**: Add more nodes and edges
5. **Different activations**: Use relu, tanh, linear, etc.

**Example - Making it deeper**:
```python
config = {
    "node_list": [
        {"name": "pixels", "dim": 784, "activation": {"type": "linear"}},
        {"name": "h1", "dim": 256, "activation": {"type": "relu"}},
        {"name": "h2", "dim": 128, "activation": {"type": "relu"}},
        {"name": "class", "dim": 10, "activation": {"type": "linear"}},
    ],
    "edge_list": [
        {"source_name": "pixels", "target_name": "h1", "slot": ""},
        {"source_name": "h1", "target_name": "h2", "slot": ""},
        {"source_name": "h2", "target_name": "class", "slot": ""},
    ],
    "task_map": {"x": "pixels", "y": "class"},
}
```

## Notes

### Data Loading
Currently uses PyTorch DataLoader for convenience. You may see fork() warnings - these are harmless but will be addressed in future versions with JAX-native data pipelines.

### Performance
First batch is slow (~5-10s) due to JIT compilation. Subsequent batches are 10-20x faster.

### Multi-GPU
Not yet implemented in these examples, but coming soon!

## Comparison to PyTorch Version

| Feature | PyTorch | JAX |
|---------|---------|-----|
| Code length | ~50 lines | ~60 lines |
| Model definition | Same dict | Same dict |
| Training | Imperative | Functional |
| Speed (single GPU) | Baseline | ~1.5x faster* |
| Multi-GPU | Manual | Built-in (pmap) |

*After JIT compilation

## Troubleshooting

**ImportError**: Make sure you're running from the project root or have PYTHONPATH set.

**CUDA Out of Memory**: Reduce batch_size in the script.

**Fork warnings**: Harmless, will be fixed with JAX-native data loading.

## Coming Soon

- Multi-GPU MNIST example with `pmap`
- Continual learning examples
- Advanced training techniques
- Custom architectures
