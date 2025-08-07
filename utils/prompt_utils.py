import torch

def register_prompt_tokens(tokenizer, model=None, prompt_tokens=["The scenario is", "<newline>"]):
    """
    Register special prompt tokens into tokenizer (and optionally resize model embeddings).

    Args:
        tokenizer: A HuggingFace tokenizer (e.g., T5Tokenizer)
        model: (optional) Model to resize embeddings after adding tokens
        prompt_tokens: List of prompt tokens to register

    Returns:
        token_ids: Dict mapping from token string to token ID
    """
    newly_added = []

    # 🔍 Check which prompt tokens are not yet registered
    for token in prompt_tokens:
        if token not in tokenizer.additional_special_tokens:
            newly_added.append(token)

    # ➕ Register new special tokens and optionally resize model embeddings
    if newly_added:
        tokenizer.add_special_tokens({'additional_special_tokens': newly_added})
        print(f"✅ Registered special tokens: {newly_added}")

        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
            print(f"✅ Resized model embeddings to {len(tokenizer)} tokens.")
    else:
        print("🔹 All tokens already registered.")

    # 🔢 Print token IDs for verification
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in prompt_tokens}
    for token, tid in token_ids.items():
        print(f"🆔 Token ID for '{token}': {tid}")

    return token_ids
