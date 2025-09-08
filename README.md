# Linear Recurrent Units in Tensorflow: An Unofficial Implementation
This repository presents an unofficial implementation of Linear Recurrent Units (LRUs) proposed by Google DeepMind, utilizing Tensorflow. LRUs draw inspiration from Deep State-Space Machines, with a particular focus on S4 and S5 models.

# Installation:
```
$ pip install LRU-tensorflow
```
# Usage:
```python
import tensorflow as tf
from LRU_tensorflow import LRU

# Define model dimensions
state_features = 19  # N: state dimension
input_size     = 5   # H: model (input) dimension

# Create LRU layer
lru = LRU(N=state_features, H=input_size)

# Prepare sample input: (batch_size, seq_length, input_size)
batch_size = 32
seq_length = 50
test_input = tf.random.uniform((batch_size, seq_length, input_size))

# Forward pass
predictions = lru(test_input)

# Output shape: (batch_size, seq_length, input_size)
print(predictions.shape)
# Expected: (32, 50, 5)
```

# Paper:
<a href='https://arxiv.org/abs/2303.06349'>Resurrecting Recurrent Neural Networks for Long Sequences</a>


# Notes:
+ If you require an implementation that supports 3-dimensional input sequences, you may want to refer to <a href='https://github.com/Gothos/LRU-pytorch'>github.com/Gothos/LRU-pytorch</a>. However, please be aware that this alternative implementation might be slower due to the absence of associative scans.
