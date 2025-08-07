import gradio as gr
import os
import glob
import json

def create_graph_rating_app(image_dir, save_path="human_graph_feedback.json", beam_width=5):

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    assert len(image_files) > 0, f"No images found in {image_dir}!"

    def interface(*scores):
        feedback = []
        for img_path, score in zip(image_files, scores):
            feedback.append({
                "image": os.path.basename(img_path),
                "score": int(score)
            })

        # 保存到json
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)

        return f"✅ Feedback saved to {save_path}!"

    inputs = []
    for img_path in image_files:
        with gr.Row():
            inputs.append(gr.Image(value=img_path, label=os.path.basename(img_path), interactive=False))
            inputs.append(gr.Radio(choices=[1,2,3,4,5], label="Rating (1~5)", value=3))

    # create Gradio interface
    demo = gr.Interface(
        fn=interface,
        inputs=[inp for i, inp in enumerate(inputs) if i%2==1],  # 只拿评分器
        outputs="text",
        title="Graph Feedback Collection",
        description=f"Please rate each generated graph from 1 (bad) to 5 (good). Total {len(image_files)} graphs."
    )

    return demo
