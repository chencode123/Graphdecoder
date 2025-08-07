import gradio as gr
import os
import glob
import json

def create_graph_rating_from_json(metadata_json_path, image_dir, save_path="human_graph_feedback.json"):
    """
    Args:
        metadata_json_path: Path to your beam_graphs_metadata.json
        image_dir: Directory containing the images
        save_path: Where to save human feedback
    """

    # load metadata json
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert len(metadata) > 0, "No entries in metadata!"

    def submit_feedback(*scores):
        scores = list(scores)  # 将 tuple 转为 list
        feedback = []
        for entry, score in zip(metadata, scores):
            human_score = int(score) if score is not None else -1
            feedback.append({
                "beam_idx": entry["beam_idx"],
                "image_file": entry["image_file"],
                "text": entry["text"],
                "log_score": entry["log_score"],
                "probability": entry["probability"],
                "human_score": human_score
            })
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)
        return f"✅ Saved {len(scores)} ratings to {save_path}!"

    with gr.Blocks() as demo:
        gr.Markdown(f"## Rate Generated Graphs\nTotal: {len(metadata)} graphs. Please rate carefully.")
        
        score_components = []

        for entry in metadata:
            img_path = os.path.join(image_dir, entry["image_file"])
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(value=img_path, label=entry["image_file"], height=300)
                with gr.Column(scale=1):
                    gr.Markdown(f"**Probability:** {entry['probability']:.6f}")
                    score = gr.Radio(choices=list(range(0,11)), label="Rating (1=bad, 10=good)", value=5)
                    score_components.append(score)

        submit_btn = gr.Button("Submit Feedback")
        output_text = gr.Textbox(label="Result")

        submit_btn.click(fn=submit_feedback, inputs=score_components, outputs=output_text)

    return demo