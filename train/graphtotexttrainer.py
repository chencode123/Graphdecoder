import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from tqdm import tqdm, trange
import dgl
import wandb

from data.graphtotextdataset import GraphToTextDataset
from model.graphtotextmodel import GraphToTextModel
from nltk.translate.bleu_score import corpus_bleu

class GraphToTextTrainer:
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
        
        self.model = GraphToTextModel(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            vocab_size=len(self.tokenizer)
        )
        self.model.tokenizer = self.tokenizer
        self.model.to(self.device)
        # ✅ set up tokenizer and prompt_token_id
        self.model.prompt_token_id = self.tokenizer.convert_tokens_to_ids("The scenario is")
        if weight_path and os.path.exists(weight_path):
            print(f"✅ Loaded model weights from: {weight_path}")
            self.model.load_state_dict(torch.load(weight_path, map_location=device))

    def collate_fn(self, batch, max_length=128):
        """
        Collate function to batch graph-text pairs.
        """
        graphs, texts = zip(*batch)
        batched_graph = dgl.batch(graphs)

        tokens = self.tokenizer(
            list(texts),
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = input_ids.clone()

        return batched_graph, input_ids, attention_mask, labels

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

        for step, (batched_graph, input_ids, attention_mask, labels) in enumerate(progress_bar):
            batched_graph = batched_graph.to(self.device)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            loss = self.model(batched_graph, labels=labels)

            if torch.isnan(loss):
                print("⚠️ NaN loss at step", step)
                continue

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
        graphs_train,
        texts_train,
        epochs,
        batch_size,
        lr,
        save_path=None,
        graphs_val=None,
        texts_val=None,
        val_every = None
    ):
        """
        Full training loop.

        :param graphs: List of DGLGraph objects
        :param texts: List of target texts
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param lr: Learning rate
        :param save_path: Path to save best model
        :return: Trained model
        """
        dataset = GraphToTextDataset(graphs_train, texts_train)
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
            # evaluate BLEU every 20 epochs
                if graphs_val is not None and texts_val is not None and epoch % val_every == 0:
                    val_dataset = list(zip(graphs_val, texts_val))
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=lambda batch: (
                            dgl.batch([g for g, _ in batch]),
                            [t for _, t in batch]
                        )
                    )

                    all_preds = []
                    all_refs = []

                    for bg, text_refs in tqdm(val_loader, desc=f"🧪 Validating (Epoch {epoch})", ncols=100):
                        bg = bg.to(self.device)
                        preds = self.model.generate(bg)
                        decoded = [beam[0][0] if beam else "" for beam in preds]
                        all_preds.extend(decoded)
                        all_refs.extend(text_refs)

                    hypotheses = [pred.split() for pred in all_preds]
                    references = [[ref.split()] for ref in all_refs]
                    bleu = corpus_bleu(references, hypotheses)
                    print(f"\n🧪 Epoch {epoch}: BLEU = {bleu:.4f}")
                    wandb.log({"val_bleu": bleu})

        return self.model
