import os

from peft import PeftModel
from safetensors.torch import load_file


def load_model_weights(
    model,
    pretrained_path: str = None,
    use_lora: bool = False,
    is_trainable: bool = False,
):
    if not pretrained_path:
        return model

    if use_lora:
        return PeftModel.from_pretrained(model, pretrained_path, is_trainable=is_trainable)

    checkpoint_path = os.path.join(pretrained_path, "model.safetensors")
    state_dict = load_file(checkpoint_path)
    state_dict.pop("text_embedding.weight", None)
    model_state_dict = model.state_dict()
    for key in ("classifier.weight", "classifier.bias"):
        if key in state_dict and key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    return model
