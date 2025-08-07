import os
import json

def collect_human_feedback(generated, save_file="results/human_feedback.json", score_range=(1, 5)):
    """
    Collect human feedback for generated outputs.

    Args:
        generated: List[List[(text, log_score)]], shape (batch_size x beam_size)
        save_file: Path to save collected feedback
        score_range: Tuple indicating the valid range for human scores (inclusive)
    """

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    feedback_list = []

    min_score, max_score = score_range

    print("\n📋 Starting human feedback collection...")
    print(f"📝 Please rate each output between {min_score} and {max_score} (inclusive).")

    for batch_idx, beam_list in enumerate(generated):
        print(f"\n==== Batch {batch_idx} ====")
        for beam_idx, (text, log_score) in enumerate(beam_list):
            print(f"\n🔢 Beam {beam_idx+1}")
            replaced_text = text.replace('<newline>', '\n')
            print(f"📝 Text:\n{replaced_text}")

            # human score
            while True:
                try:
                    human_score = int(input(f"🌟 Please rate this output ({min_score}~{max_score}): "))
                    if min_score <= human_score <= max_score:
                        break
                    else:
                        print(f"⚠️  Please enter an integer between {min_score} and {max_score}.")
                except ValueError:
                    print("⚠️  Invalid input. Please enter an integer number.")

            feedback_list.append({
                "batch_idx": batch_idx,
                "beam_idx": beam_idx,
                "text": text,
                "log_score": log_score,
                "human_score": human_score
            })

    # save
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Human feedback collection completed and saved to {save_file}!")
