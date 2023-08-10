# Linear Recurrent Units in Tensorflow: An Unofficial Implementation
This repository presents an unofficial implementation of Linear Recurrent Units (LRUs) proposed by Google DeepMind, utilizing Tensorflow. LRUs draw inspiration from Deep State-Space Machines, with a particular focus on S4 and S5 models.

# Notes:
+ If you require an implementation that supports 3-dimensional input sequences, you may want to refer to <a href='https://github.com/Gothos/LRU-pytorch'>github.com/Gothos/LRU-pytorch</a>. However, please be aware that this alternative implementation might be slower due to the absence of associative scans.

# Installation:
```
$ pip install LRU-tensorflow
```
# Usage:
```python
import tensorflow

from LRU_tensorflow import LRU

lru = LRU(N=state_features, H=input_size) 
test_input = tensorflow.random.uniform(batch_size, seq_length, input_size)  # Example Test Input
predictions = lru(test_input) # Get predictions
```

# Paper:
<a href='https://arxiv.org/abs/2303.06349'>Resurrecting Recurrent Neural Networks for Long Sequences</a>
