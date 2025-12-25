# Predictive Coding Examples

This directory contains example scripts demonstrating FabricPC.

## Quick Start

```bash
# Install FabricPC in editable mode (from project root)
pip install -e ".[dev,torch,viz]"

# Run MNIST demo
python examples/mnist_demo.py
```

## Examples

### `mnist_demo.py`

**Description**: MNIST classification example demonstrating the basic PC workflow.

**Architecture**:
- Input layer: 784 units (28x28 flattened images)
- Hidden layer 1: 256 units (sigmoid activation)
- Hidden layer 2: 64 units (sigmoid activation)
- Output layer: 10 units (class logits)

**Configuration**:
- Optimizer: Adam (lr=1e-3)
- Inference: 20 steps @ eta=0.05
- Training: 20 epochs, batch size 200

**Expected Results**:
```
Epoch 1/20, Loss: 0.4529
Epoch 2/20, Loss: 0.1903
Epoch 3/20, Loss: 0.0757
Epoch 4/20, Loss: 0.0482
...
Epoch 18/20, Loss: 0.0052
Epoch 19/20, Loss: 0.0047
Epoch 20/20, Loss: 0.0043
Avg Training time: 2.05 seconds per epoch

Evaluating...
Test Accuracy: 97.93%
Test Loss: 0.0349
```

### transformer_demo.py

Using Backpropagation training method (6-block Transformer on Shakespeare dataset)

  [DEBUG] per-token CE loss: min=-0.0000, max=11.7602, mean=1.4054

  [DEBUG] per-token intrinsic perplexity: min=1.0000, max=25.2567, mean=3.3277

  [DEBUG] prob of correct token: min=0.000008, max=1.000000, mean=0.593641

  [DEBUG] batch loss: 1.4054

Test - Loss: 1.9770, Perplexity: 7.22, Acc: 0.5202

Epoch 1/1, Loss: 1.2071, Perplexity: 3.34

Training completed in 347.4s (347.4s per epoch)

Generating sample text...

- Prompt: 'Know, Rome, that'
----------------------------------------
Know, Rome, that is not the devils;

----------------------------------------
- Prompt: 'MENENIUS:'
----------------------------------------
MENENIUS:
The good my lord.
----------------------------------------

Using Predictive Coding training method

  [DEBUG] per-token CE loss: min=-0.0000, max=23.0259, mean=22.4215

  [DEBUG] per-token intrinsic perplexity: min=1.0000, max=1.0000, mean=1.0000

  [DEBUG] prob of correct token: min=0.000000, max=1.000000, mean=0.026245

  [DEBUG] batch loss: 22.4215

Test - Loss: 22.0098, Perplexity: 3620162304.00, Acc: 0.0441

Train Epoch 1/1, Energy: 238896016419357371257585664.0000, Loss: 21.8046, Perplexity: 2948480768.00

Training completed in 815.3s (815.3s per epoch)

PC inference phase is very sensitive to weight initialization and hyperparameters. Mode collapse in PC training but trains successfully with BP.



## Notes

### Data Loading Warning

You may see warnings about `os.fork()` when using PyTorch DataLoader:
```
RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code,
and JAX is multithreaded, so this will likely lead to a deadlock.
```

This is harmless for now but will be addressed in future versions by switching to JAX-native data pipelines.

### Performance

First batch will be slow due to JIT compilation (~5-10 seconds). Subsequent batches are fast.

## Troubleshooting
**Import Error**: Make sure FabricPC is installed:
```bash
pip install -e ".[dev,torch,viz]"
```
