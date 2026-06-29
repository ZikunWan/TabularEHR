def get_candidate_texts(task_info: dict) -> list[str]:
    task_type = task_info.get("task_type")
    if "candidate" in task_info:
        return [str(candidate) for candidate in task_info["candidate"]]
    if task_type == "binary_classification":
        return ["no", "yes"]
    if task_type == "multi_class_classification":
        return [str(idx) for idx in range(int(task_info["num_classes"]))]
    raise ValueError(f"Unsupported candidate task_type: {task_type}")


def build_candidate_embedding_texts(query_key: str, query_text: str, candidate_texts: list[str]) -> dict[str, str]:
    texts = {query_key: query_text}
    for candidate in candidate_texts:
        candidate_key = f"{query_key}:candidate:{candidate}"
        texts[candidate_key] = candidate
    return texts


def candidate_embedding_keys(query_key: str, candidate_texts: list[str]) -> list[str]:
    return [f"{query_key}:candidate:{candidate}" for candidate in candidate_texts]
