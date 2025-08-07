import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Embedding
from transformers import T5Tokenizer
from .mlp import MLP
from .positional_encoding import PositionalEncoding

class GraphToTextModel(nn.Module):
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

    def forward(self, g: dgl.DGLGraph, labels: torch.Tensor = None):
        h = g.ndata["h"]
        h = self.gnn(g, h)
        g.ndata["h"] = h

        batch_size = g.batch_size
        num_nodes = g.num_nodes() // batch_size
        hidden_dim = h.shape[-1]

        graph_embeddings = h.view(batch_size, num_nodes, hidden_dim)
        memory = self.graph_to_decoder(graph_embeddings)  # [B, S, H]

        # Prepare target tokens
        tgt = labels[:, :-1]
        tgt_y = labels[:, 1:]

        tgt_len = tgt.size(1)
        tgt_emb = self.embedding(tgt).transpose(0, 1)  # [T, B, H]
        tgt_emb = self.pos_encoder(tgt_emb)

        memory = memory.transpose(0, 1)  # [S, B, H]

        # Create causal mask for autoregressive decoding
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask
        )  # [T, B, H]

        logits = self.out_proj(output)  # [T, B, V]

        # Compute loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fn(logits.transpose(1, 2), tgt_y.transpose(0, 1))

        return loss

    def generate(self, g: dgl.DGLGraph, max_length: int = 128, beam_width: int = 3) -> list[str]:
        """
        Generates text sequences from the input graph using beam search decoding.
        """
        self.eval()
        device = next(self.parameters()).device

        h = g.ndata["h"].to(device)
        g = g.to(device)
        h = self.gnn(g, h)
        g.ndata["h"] = h

        batch_size = g.batch_size
        num_nodes = g.num_nodes() // batch_size
        hidden_dim = h.shape[-1]

        graph_embeddings = h.view(batch_size, num_nodes, hidden_dim)
        memory = self.graph_to_decoder(graph_embeddings).transpose(0, 1)  # [S, B, H]

        start_id = self.prompt_token_id
        eos_id = self.tokenizer.eos_token_id

        generated_sequences = []

        for b in range(batch_size):
            # Initialize beam with start token
            beams = [(torch.tensor([[start_id]], device=device), 0.0, False)]  # (tokens, log_prob, finished)

            for step in range(max_length):
                candidates = []

                for seq, score, is_finished in beams:
                    if is_finished:
                        candidates.append((seq, score, True))
                        continue

                    tgt_emb = self.embedding(seq).transpose(0, 1)  # [T, 1, H]
                    tgt_emb = self.pos_encoder(tgt_emb)
                    seq_len = tgt_emb.size(0)
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

                    decoder_out = self.transformer_decoder(
                        tgt_emb,
                        memory[:, b:b+1, :],
                        tgt_mask=causal_mask
                    )

                    logits = self.out_proj(decoder_out)[-1, 0]  # [V]
                    log_probs = F.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_ids = log_probs.topk(beam_width)

                    for log_prob, token_id in zip(topk_log_probs, topk_ids):
                        new_seq = torch.cat([seq, token_id.view(1, 1)], dim=1)
                        new_score = score + log_prob.item()
                        finished_flag = token_id.item() == eos_id
                        candidates.append((new_seq, new_score, finished_flag))

                # Select top-k beams by score
                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]

                # Stop if all beams are finished
                if all(f for _, _, f in beams):
                    break

            # Decode beams to text
            beam_texts = []
            for seq, score, _ in beams:
                ids = seq.squeeze(0).tolist()
                if eos_id in ids:
                    ids = ids[:ids.index(eos_id)]
                text = self.tokenizer.decode(ids, skip_special_tokens=False).strip()
                beam_texts.append((text, score))

            generated_sequences.append(beam_texts)
        return generated_sequences
