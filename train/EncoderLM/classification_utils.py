from typing import Callable, Optional

from datasets import Dataset

from common import apply_template


def normalize_label(raw_label) -> str:
    label = str(raw_label).strip()
    while len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
        label = label[1:-1].strip()

    lowered = label.lower()
    if lowered in {"yes", "y", "true"}:
        return "yes"
    if lowered in {"no", "n", "false"}:
        return "no"
    return lowered


def augment_binary_aliases(label_to_id):
    aliases = {}
    if "0" in label_to_id and "1" in label_to_id:
        aliases.update(
            {
                "no": label_to_id["0"],
                "false": label_to_id["0"],
                "n": label_to_id["0"],
                "yes": label_to_id["1"],
                "true": label_to_id["1"],
                "y": label_to_id["1"],
            }
        )
    if "no" in label_to_id and "yes" in label_to_id:
        aliases.update(
            {
                "0": label_to_id["no"],
                "false": label_to_id["no"],
                "n": label_to_id["no"],
                "1": label_to_id["yes"],
                "true": label_to_id["yes"],
                "y": label_to_id["yes"],
            }
        )
    label_to_id.update(aliases)


def tokenize_classification_dataset(
    source_dataset,
    tokenizer,
    model_name_or_path: str,
    max_seq_length: int,
    system_prompt: str,
    label_to_id,
    task_name: str,
    dataset_name: str,
    label_normalizer: Optional[Callable[[object], str]] = None,
):
    rows = []
    for index in range(len(source_dataset)):
        sample = source_dataset[index]
        instruction = sample.get("instruction", sample.get("task_info", {}).get("instruction", ""))
        prompt = apply_template(
            model_name_or_path=model_name_or_path,
            processor_or_tokenizer=tokenizer,
            input_text=str(sample.get("input", "")),
            instruction=instruction,
            system_prompt=system_prompt,
            output_text=None,
        )
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            return_token_type_ids=True,
        )
        raw_label = sample["output"]
        label = label_normalizer(raw_label) if label_normalizer is not None else str(raw_label)
        if label not in label_to_id:
            raise ValueError(
                f"Unexpected label '{raw_label}' (normalized: '{label}') "
                f"for {dataset_name} task '{task_name}'."
            )
        tokenized["labels"] = label_to_id[label]
        tokenized["idx"] = index
        rows.append(tokenized)
    return Dataset.from_list(rows)
