import wandb
from train.graphtotexttrainer import GraphToTextTrainer
import os
import sys
import torch
import pandas as pd
from transformers.modeling_outputs import BaseModelOutput
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import math
from sklearn.model_selection import train_test_split
from utils.prompt_utils import register_prompt_tokens
from nltk.translate.bleu_score import corpus_bleu

project_dir = "/home/chenyangtamu/graphdecoder/graphtoseq_3_nodes_no_equipment"
sys.path.append(project_dir)
os.chdir(project_dir)

# 📄 Load scenario text and node embeddings
cstr_combined = pd.read_csv("data/dataset/cstr_combined.csv")
beam_width=1
max_length = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📦 Load DGL graph list (take first 7600)
graphs_list = dgl.load_graphs('data/dataset/graph_data.bin')
graphs = graphs_list[0][:7600]

# 📝 Prepare output texts for each graph (used as targets during training)
scenario_texts = []
N = len(graphs)
for i in range(N):
    scenario_texts.append("The scenario is " + cstr_combined["scenario"].iloc[i])

import torch
from transformers import T5Tokenizer

# Initialize wandb
wandb.init(
    project="graph-to-text_one_node",
    name="t5_gnn_training", 
    config={
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "model": "T5-small",
        "gnn": "GINConv",
        "weight_path": "weights/train_3_nodes_no_equipment.pth", # or a path if you have pretrained weights
        "save_path": "weights/train_3_nodes_no_equipment.pth"
    }
)

# train: valid: test == 8:1:1
graphs_train_val, graphs_test, texts_train_val, texts_test = train_test_split(
    graphs,
    scenario_texts,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

graphs_train, graphs_val, texts_train, texts_val = train_test_split(
    graphs_train_val,
    texts_train_val,
    test_size=0.125,  
    random_state=42,
    shuffle=True
)

# Initialize the trainer
trainer = GraphToTextTrainer(
    node_dim=512, 
    tokenizer_path="weights/saved_tokenizer_with_prompt", 
    device="cuda", 
    weight_path=wandb.config.weight_path
)

# Start training
trained_model = trainer.train(
    graphs_train=graphs_train,
    texts_train=texts_train,
    graphs_val = graphs_val,
    texts_val = texts_val,
    epochs=wandb.config.epochs,
    batch_size=wandb.config.batch_size,
    lr=wandb.config.learning_rate,
    save_path=wandb.config.save_path,
    val_every = 100 # run validaiton every val_every epochs
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.graphtotextmodel import GraphToTextModel
# 1. Load the tokenizer used during training 
tokenizer = T5Tokenizer.from_pretrained("weights/saved_tokenizer_with_prompt")  # ← Path to the tokenizer saved during training

# 2. Initialize the model (vocabulary size must match the tokenizer)
trained_model = GraphToTextModel(
    node_dim=512,
    hidden_dim=512,
    vocab_size=len(tokenizer),
    max_len=128
)
trained_model.prompt_token_id = tokenizer.convert_tokens_to_ids("The scenario is")

# 3. Set tokenizer and prompt_token_id
trained_model.tokenizer = tokenizer
# 5. Load the trained model parameters (ensure vocabulary size matches)
trained_model.load_state_dict(torch.load("weights/train_3_nodes_no_equipment.pth", map_location="cuda"))
# 6. Move model to device
trained_model.to(device)

print("\n🎯 Evaluating on test set...")

tokenizer = T5Tokenizer.from_pretrained("weights/saved_tokenizer_with_prompt")
trained_model.tokenizer = tokenizer
trained_model.prompt_token_id = tokenizer.convert_tokens_to_ids("The scenario is")

batch_size = 128  

from torch.utils.data import DataLoader
from tqdm import tqdm

def dgl_collate_fn(batch):
    return dgl.batch(batch)

test_loader = DataLoader(
    graphs_test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=dgl_collate_fn  
)

test_predictions = []

test_predictions = []
for batched_graph in tqdm(test_loader, desc="Generating", ncols=100):
    batched_graph = batched_graph.to(device)
    outputs = trained_model.generate(batched_graph, beam_width=beam_width)
    test_predictions.extend([beam[0][0] for beam in outputs])  # ✅ 只取 top-1 文本

test_references = [[ref.split()] for ref in texts_test]
test_hypotheses = [pred.split() for pred in test_predictions]

test_bleu = corpus_bleu(test_references, test_hypotheses)
print(f"📊 Final BLEU score on test set: {test_bleu:.4f}")
wandb.log({"test_bleu_score": float(test_bleu)})

import time
time.sleep(2)
wandb.finish()

