from transformers import T5Tokenizer, T5Config
import torch
import os
import sys
from model.graphtotextmodel import GraphToTextModel
from model.humanupdatemodel import HumanGraphToTextModel
import dgl
import pandas as pd
from utils.visualization import draw_topk_beam_graphs
from utils.human_feedback import collect_human_feedback
import wandb
from train.human_train import HumanGraphToTextTrainer
from data.human_feedback_utils import load_human_feedback
from utils.gradio_graph_feedback import create_graph_rating_app
from data.text_generator import generate_scenario_texts
from utils.gradio_graph_feedback_blocks import create_graph_rating_from_json

project_dir = "/home/chenyangtamu/graphdecoder/graphtoseq_3_nodes_no_equipment"
sys.path.append(project_dir)
os.chdir(project_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

#set the output number of the figure
beam_width=5
num = 0
max_length = 128
# 📄 Load scenario text and node embeddings
node2_combined = pd.read_csv("data/case_study/node2_combined.csv")
graphs_list = dgl.load_graphs('data/case_study/graph_data.bin')
graphs = graphs_list[0][:2]
scenario_texts = generate_scenario_texts(graphs, node2_combined)

# 1. Load the tokenizer used during training (recommended to use your own saved one)
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
# 7. Prepare the graph input
g = graphs[num].to(device)

####################8. Generate text
trained_model.eval()
with torch.no_grad():
    generated = trained_model.generate(g, beam_width=beam_width)

# g_cleaned = generated[0].removeprefix("The scenario is").strip()
g_true = scenario_texts[num].removeprefix("The scenario is").strip()
g_cleaned = generated[0][0][0].removeprefix("The scenario is").strip()

# 9. Output
# print("📝 Generated:\n", g_cleaned)
print("✅ Ground Truth:\n", g_true)
print("✅ generated:\n", g_cleaned)

draw_topk_beam_graphs(generated, topk=beam_width, save_dir="results/beam_graphs")

# launch grading panel
demo = create_graph_rating_from_json(
    metadata_json_path = "results/beam_graphs/beam_graphs_metadata.json",
    image_dir="results/beam_graphs",
    save_path="results/human_feedback/human_graph_feedback.json"
)
demo.launch(share=True)

##############Huamn feedback

# # Initialize wandb
# wandb.init(
#     project="graph-to-text_one_node",
#     name="t5_gnn_training",
#     config={
#         "epochs": 100,
#         "batch_size": 8,
#         "learning_rate": 1e-4,
#         "model": "T5-small",
#         "gnn": "GINConv",
#         "weight_path": "weights/train_518.pth",  # or a path if you have pretrained weights
#         "save_path": "weights/human.pth"
#     }
# )

# texts_list, weights_list = load_human_feedback("results/human_feedback/human_graph_feedback.json")

# # Initialize the trainer
# trainer_human = HumanGraphToTextTrainer(
#     node_dim=512, 
#     tokenizer_path="weights/saved_tokenizer_with_prompt", 
#     device="cuda", 
#     weight_path=wandb.config.weight_path
# )

# # Start training
# trained_model_human = trainer_human.train(
#     graphs=[g],
#     texts=texts_list,
#     human_weights=weights_list,
#     epochs=wandb.config.epochs,
#     batch_size=wandb.config.batch_size,
#     lr=wandb.config.learning_rate,
#     save_path=wandb.config.save_path
# )

# # Finish wandb logging
# wandb.finish()

# trained_model.load_state_dict(torch.load("weights/human.pth", map_location="cuda"))
# # 6. Move model to device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trained_model.to(device)

# # 7. Prepare the graph input
# # 8. Generate text

# trained_model.eval()
# with torch.no_grad():
#     generated = trained_model.generate(g, beam_width=beam_width)

# g_true = scenario_texts[num].removeprefix("The scenario is").strip()
# g_cleaned = generated[0][0][0].removeprefix("The scenario is").strip()

# # 9. Output
# # print("📝 Generated:\n", g_cleaned)
# print("✅ Ground Truth:\n", g_true)
# print("✅ generated:\n", g_cleaned)

# draw_topk_beam_graphs(generated, topk=beam_width, save_dir="results/beam_graphs_human")

# from utils.gradio_graph_feedback_blocks import create_graph_rating_from_json

# # 启动打分界面
# demo = create_graph_rating_from_json(
#     metadata_json_path = "results/beam_graphs/beam_graphs_metadata.json",
#     image_dir="results/beam_graphs_human",
#     save_path="results/human_feedback/human_graph_feedback2.json"
# )
# demo.launch(share=True)