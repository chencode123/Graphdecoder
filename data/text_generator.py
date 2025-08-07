# scenario_utils/text_generator.py

def generate_scenario_texts(graphs, cstr_combined):
    """
    Generate output scenario texts from graphs and scenario DataFrame.

    Args:
        graphs (list): List of graph data (e.g., DGLGraph, PyG Graph, etc.).
        cstr_combined (pd.DataFrame): DataFrame containing a 'scenario' column.

    Returns:
        list: List of scenario texts in the format "The scenario is ..."
    """
    scenario_texts = []
    N = len(graphs)
    for i in range(N):
        scenario_texts.append("The scenario is " + cstr_combined["scenario"].iloc[i])
    return scenario_texts
