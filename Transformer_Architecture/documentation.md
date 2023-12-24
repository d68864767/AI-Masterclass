# Transformer Architecture Documentation

## Overview

The Transformer architecture is a type of model architecture used in the field of deep learning and more specifically, in the domain of Natural Language Processing (NLP). It was introduced in the paper "Attention is All You Need" by Vaswani et al. and has since been the foundation for many state-of-the-art models such as GPT and BERT.

The Transformer model is based on self-attention mechanisms and does not require sequence-aligned inputs like recurrent neural networks (RNNs). This makes it more parallelizable and faster to train.

## Code Explanation

The provided code in `Transformer_Architecture/code.py` implements a Transformer model using PyTorch. Here's a breakdown of the code:

### Importing Libraries

The necessary libraries for creating the Transformer model are imported. This includes PyTorch and its submodules.

```python
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
```

### Defining Constants

Several constants are defined for the Transformer model, including the embedding size, the number of heads in the multiheadattention models, the number of sub-encoder-layers in the transformer encoder, the dropout value, and the device to run the model on.

```python
EMBEDDING_SIZE = 512
NHEAD = 8
NUM_LAYERS = 6
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Transformer Model Class

A class `TransformerModel` is defined which inherits from the `nn.Module` class of PyTorch. This class represents the Transformer model.

The `__init__` method initializes the model layers including the embedding layer, the positional encoding layer, the transformer encoder layer, and the final linear layer.

The `forward` method defines the forward pass of the model. It takes in the input text and passes it through the embedding layer, applies the positional encoding, passes it through the transformer encoder, and finally through the linear layer.

```python
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        ...
    def forward(self, src, src_mask):
        ...
```

## Tutorial

For a more detailed walkthrough of the Transformer architecture and a tutorial on how to use this code, please refer to the `Transformer_Architecture/tutorial.md` file.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).
