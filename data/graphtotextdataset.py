from torch.utils.data import Dataset

class GraphToTextDataset(Dataset):
    def __init__(self, graphs, texts):
        """
        Custom Dataset for graph-to-text generation tasks.

        :param graphs: List of DGLGraph objects representing input graphs
        :param texts: List of target textual descriptions (one per graph)
        """
        self.graphs = graphs
        self.texts = texts

    def __len__(self):
        """
        :return: Total number of graph-text pairs
        """
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        :param idx: Index of the sample
        :return: A tuple (graph, text) at the given index
        """
        return self.graphs[idx], self.texts[idx]
