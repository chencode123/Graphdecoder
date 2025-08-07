from torch.utils.data import Dataset
import torch
import dgl

class HumanFeedbackBeamDataset(Dataset):
    def __init__(self, graphs, list_of_texts, list_of_weights, tokenizer, max_len=128):
        """
        graphs: list of DGLGraph, length = N
        list_of_texts: list of list, each graph has beam_size texts
        list_of_weights: list of list, each graph has beam_size human scores (normalized)
        tokenizer: Huggingface tokenizer
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for g, texts, weights in zip(graphs, list_of_texts, list_of_weights):
            for text, weight in zip(texts, weights):
                self.samples.append((g, text, weight))  # 展开成单样本

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g, text, weight = self.samples[idx]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        labels = encoded.input_ids.squeeze(0)  # [seq_len]

        return g, labels, weight
