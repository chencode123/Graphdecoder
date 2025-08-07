import json
from collections import defaultdict

def load_human_feedback(json_path, score_range=(0, 10), min_weight=0.1):
    """
    Load and process human feedback from JSON file.

    Args:
        json_path: Path to human_feedback.json file
        score_range: Tuple indicating (min_score, max_score) for normalization
        min_weight: Minimum allowed normalized weight

    Returns:
        texts_list: List of list of texts per graph
        weights_list: List of list of normalized weights per graph
    """
    with open(json_path, "r", encoding="utf-8") as f:
        feedback_data = json.load(f)

    texts_per_graph = defaultdict(list)
    scores_per_graph = defaultdict(list)

    for item in feedback_data:
        beam_idx = item['beam_idx']
        texts_per_graph[beam_idx].append(item['text'])
        scores_per_graph[beam_idx].append(item['human_score'])

    texts_list = [texts_per_graph[i] for i in sorted(texts_per_graph.keys())]
    scores_list = [scores_per_graph[i] for i in sorted(scores_per_graph.keys())]

    # Normalize scores to weights
    min_score, max_score = score_range
    weights_list = [
        [(score - min_score) / (max_score - min_score) for score in score_list]
        for score_list in scores_list
    ]

    return texts_list, weights_list
