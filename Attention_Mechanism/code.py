```python
# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define constants
EMBEDDING_SIZE = 512  # Embedding size
NHEAD = 8  # the number of heads in the multiheadattention models
DROPOUT = 0.1  # the dropout value
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        assert (
            self.head_dim * num_heads == embedding_size
        ), "Embedding size needs to be divisible by num heads"

        self.values_fc = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys_fc = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries_fc = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embedding_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values_fc(values)
        keys = self.keys_fc(keys)
        queries = self.queries_fc(query)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, num_heads, head_dim),
        # keys shape: (N, key_len, num_heads, head_dim)
        # energy: (N, num_heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embedding_size ** (1 / 2)), dim=3)
        # attention shape: (N, num_heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )
        # attention shape: (N, num_heads, query_len, key_len)
        # values shape: (N, value_len, num_heads, head_dim)
        # out after matrix multiply: (N, query_len, num_heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embedding_size)

        return out
```
