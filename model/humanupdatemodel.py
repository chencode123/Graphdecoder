import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Embedding
from transformers import T5Tokenizer
from .mlp import MLP
from .positional_encoding import PositionalEncoding


class HumanGraphToTextModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 512,
        vocab_size: int = None,  # Corresponds to T5 tokenizer vocab size
        max_len: int = 128,
        nhead: int = 8,
        num_decoder_layers: int = 8,
        readout: str = "mean",
    ):
        """
        Graph-to-text generation model using GNN + Transformer decoder.
        """
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size

        # Graph Neural Network: GINConv with a simple MLP
        self.gnn = GINConv(MLP(node_dim, hidden_dim), aggregator_type="sum")

        # Linear transformation from GNN output to decoder input format
        self.graph_to_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim)
        )

        # Embedding and positional encoding for decoder input
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)

        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection layer
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

        # Tokenizer (T5) used for encoding/decoding text
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        self.readout_type = readout

    def forward(self, g: dgl.DGLGraph, labels: torch.Tensor = None, human_weights: torch.Tensor = None):
        h = g.ndata["h"]
        h = self.gnn(g, h)
        g.ndata["h"] = h

        batch_size = g.batch_size
        num_nodes = g.num_nodes() // batch_size
        hidden_dim = h.shape[-1]

        graph_embeddings = h.view(batch_size, num_nodes, hidden_dim)
        memory = self.graph_to_decoder(graph_embeddings)  # [B, S, H]

        tgt = labels[:, :-1]
        tgt_y = labels[:, 1:]

        tgt_len = tgt.size(1)
        tgt_emb = self.embedding(tgt).transpose(0, 1)  # [T, B, H]
        tgt_emb = self.pos_encoder(tgt_emb)
        memory = memory.transpose(0, 1)  # [S, B, H]

        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask
        )
        logits = self.out_proj(output)  # [T, B, V]

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')
        logits = logits.transpose(0, 1)  # (B, T, V)
        raw_loss = loss_fn(logits.transpose(1, 2), tgt_y)  # logits (B, V, T), tgt_y (B, T)
        
        loss_per_sample = raw_loss.mean(dim=1)  # [B]，每个样本一个loss

        if human_weights is not None:
            assert human_weights.shape == loss_per_sample.shape, f"human_weights shape {human_weights.shape} != loss_per_sample shape {loss_per_sample.shape}"
            loss_per_sample_after = loss_per_sample * human_weights  # 权重加权

        # 最后再mean整个batch
        loss = loss_per_sample_after.mean()

        return loss

