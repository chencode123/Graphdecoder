import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from tqdm import tqdm, trange
import dgl
import wandb
from data.human_feedback_dataset import HumanFeedbackBeamDataset
from model.humanupdatemodel import HumanGraphToTextModel

class HumanGraphToTextTrainer:
    def __init__(
        self,
        node_dim=512,
        hidden_dim=512,
        tokenizer_path="weights/saved_tokenizer_with_prompt",
        device="cuda",
        weight_path=None
    ):
        """
        Trainer class for GraphToTextModel.

        :param node_dim: Input node feature dimension
        :param hidden_dim: Hidden feature dimension
        :param tokenizer_path: Path to saved T5 tokenizer
        :param device: 'cuda' or 'cpu'
        :param weight_path: Path to pre-trained model weights (optional)
        """
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

        self.model = HumanGraphToTextModel(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            vocab_size=len(self.tokenizer)
        )
        self.model.tokenizer = self.tokenizer
        self.model.to(self.device)

        if weight_path and os.path.exists(weight_path):
            print(f"✅ Loaded model weights from: {weight_path}")
            self.model.load_state_dict(torch.load(weight_path, map_location=device))

    def collate_fn(self, batch):
        """
        Collate function to batch graph-label-weight triples.
        """
        graphs, labels, human_weights = zip(*batch)

        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels, dim=0)  # (batch_size, seq_len)
        human_weights = torch.tensor(human_weights, dtype=torch.float).view(-1)  # ✅ FIXED

        return batched_graph, labels, human_weights


    def train_one_epoch(self, dataloader, optimizer, epoch):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch} - Training",
            unit="batch",
            leave=False,
            ncols=100
        )

        for step, (batched_graph, labels, human_weights) in enumerate(progress_bar):
            batched_graph = batched_graph.to(self.device)
            labels = labels.to(self.device)
            human_weights = human_weights.to(self.device)

            optimizer.zero_grad()
            loss = self.model(batched_graph, labels=labels, human_weights=human_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")
            wandb.log({"batch_loss": batch_loss})

        epoch_loss = total_loss / len(dataloader)
        wandb.log({"epoch_loss": epoch_loss})
        return epoch_loss

    def train(
        self,
        graphs,
        texts,
        human_weights,
        epochs,
        batch_size,
        lr,
        save_path=None
    ):
        """
        Full training loop with human feedback.

        :param graphs: List of DGLGraph objects
        :param texts: List of list of texts per graph (multiple beams)
        :param human_weights: List of list of human feedback weights per graph
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param lr: Learning rate
        :param save_path: Path to save best model
        :return: Trained model
        """

        # 🛠 组建Dataset
        dataset = HumanFeedbackBeamDataset(
            graphs=graphs,
            list_of_texts=[texts],
            list_of_weights=[human_weights],
            tokenizer=self.tokenizer
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: self.collate_fn(b)
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        epoch_bar = trange(1, epochs + 1, desc="Epoch Progress", ncols=100)
        best_loss = 1e9

        for epoch in epoch_bar:
            loss = self.train_one_epoch(dataloader, optimizer, epoch)
            epoch_bar.set_postfix(epoch=epoch, loss=f"{loss:.4f}")

            if loss < best_loss and save_path:
                torch.save(self.model.state_dict(), save_path)
                best_loss = loss

        return self.model

