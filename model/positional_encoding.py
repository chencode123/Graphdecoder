import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        """
        Implements the positional encoding module from the Transformer architecture.
        :param d_model: the dimensionality of embeddings
        :param dropout: dropout probability
        :param max_len: maximum sequence length supported
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix of shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the dimension
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the dimension

        pe = pe.unsqueeze(1)  # Shape becomes [max_len, 1, d_model] => [T, B=1, D]
        self.register_buffer('pe', pe)  # Register as buffer so it's saved with the model, but not trained

    def forward(self, x):
        """
        :param x: Tensor of shape [T, B, D]
        :return: Tensor of same shape with positional encoding added
        """
        x = x + self.pe[:x.size(0)]  # Add positional encoding (T-dimension aligned)
        return self.dropout(x)
