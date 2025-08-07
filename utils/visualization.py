# utils/visualization.py

import math
import os
from IPython.display import Image, display
from graphviz import Digraph
import json

def parse_edges_from_text(generated_text):
    clean_text = generated_text.replace("<newline>", "\n")
    edges = []
    for line in clean_text.strip().split("\n"):
        if "causes" in line:
            parts = line.split("causes")
            if len(parts) >= 2:
                source = parts[0].strip()
                target = "causes".join(parts[1:]).strip()
                edges.append((source, target))
    return edges

def draw_generated_graph(text, title="Generated Graph", output_file=None):
    edges = parse_edges_from_text(text)

    dot = Digraph(comment=title, format='png')
    dot.attr(dpi='300')
    dot.attr(rankdir="TB", size="8,5")  # 上下布局

    added_edges = set()
    for src, tgt in edges:
        if (src, tgt) not in added_edges:
            dot.edge(src, tgt)
            added_edges.add((src, tgt))

    if output_file:
        dot.render(output_file, format='png', cleanup=True)
        print(f"📦 Graph saved to {output_file}.png")

    return dot

def draw_topk_beam_graphs(generated, topk, save_dir="results/beam_graphs"):
    """
    Args:
        generated: List[List[(text, score)]], batch_size x beam_size
        topk: How many beams to visualize
        save_dir: where to save graph images and metadata
    """

    os.makedirs(save_dir, exist_ok=True)

    topk_list = generated[0][:topk]

    feedback_records = []  

    for i, (text, score) in enumerate(topk_list):
        cleaned_text = text.removeprefix("The scenario is").strip()
        prob = math.exp(score)

        print(f"\n🔢 Beam {i+1}")
        print(f"📝 Text: {cleaned_text}")
        print(f"📊 Log Score: {score:.2f}")
        print(f"🎯 Probability: {prob:.6f}")

        # save figure
        output_file = os.path.join(save_dir, f"beam_output_{i+1}")
        dot = draw_generated_graph(cleaned_text, title=f"Beam {i+1}", output_file=output_file)
        display(Image(filename=output_file + ".png"))

        # save text
        feedback_records.append({
            "beam_idx": i + 1,
            "image_file": f"beam_output_{i+1}.png",
            "text": cleaned_text,
            "log_score": round(score, 4),
            "probability": round(prob, 6)
        })

    # save to json
    json_save_path = os.path.join(save_dir, "beam_graphs_metadata.json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(feedback_records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Metadata saved to {json_save_path}")