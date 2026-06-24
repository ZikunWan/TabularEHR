import os
from typing import Callable, Optional, Sequence

from safetensors.torch import load_file


def _checkpoint_file(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "model.safetensors")
    return path


def _load_checkpoint_state_dict(path: str):
    checkpoint_path = _checkpoint_file(path)
    state_dict = load_file(checkpoint_path)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}, checkpoint_path


def _filter_state_dict(
    state_dict,
    include_prefixes: Optional[Sequence[str]] = None,
    exclude_prefixes: Sequence[str] = ("text_embedding.",),
):
    return {
        key: value
        for key, value in state_dict.items()
        if (include_prefixes is None or key.startswith(tuple(include_prefixes)))
        and not key.startswith(tuple(exclude_prefixes))
    }


def load_encoder_weights(
    model,
    pretrained_path: str,
    log_fn: Optional[Callable[..., None]] = None,
):
    state_dict, checkpoint_path = _load_checkpoint_state_dict(pretrained_path)
    encoder_state_dict = _filter_state_dict(
        state_dict,
        include_prefixes=("encoder.", "adapter."),
    )
    model.load_state_dict(encoder_state_dict, strict=False)
    if log_fn is not None:
        log_fn(f"Loaded {len(encoder_state_dict)} encoder tensors from {checkpoint_path}.")
    return model


def load_task_model_weights(
    model,
    checkpoint_path: str,
    fine_tune_mode: Optional[str] = None,
    trainable_module_names: Sequence[str] = ("classifier",),
    log_fn: Optional[Callable[..., None]] = None,
):
    state_dict, resolved_checkpoint_path = _load_checkpoint_state_dict(checkpoint_path)
    task_state_dict = _filter_state_dict(state_dict)
    model.load_state_dict(task_state_dict, strict=False)
    if log_fn is not None:
        log_fn(f"Loaded {len(task_state_dict)} model tensors from {resolved_checkpoint_path}.")
    if fine_tune_mode is not None:
        model = apply_fine_tune_mode(
            model,
            fine_tune_mode,
            trainable_module_names=trainable_module_names,
            log_fn=log_fn or print,
        )
    return model


def apply_fine_tune_mode(
    model,
    mode: str,
    trainable_module_names: Sequence[str] = ("classifier",),
    log_fn: Callable[..., None] = print,
):
    if mode == "full_fine_tune":
        log_fn("Fine-tune mode: full_fine_tune")
        return model

    if mode != "linear_probe":
        raise ValueError("fine_tune_mode must be 'full_fine_tune' or 'linear_probe'")

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_prefixes = tuple(f"{name}." for name in trainable_module_names)
    for name, parameter in model.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad = True

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    log_fn(
        f"Fine-tune mode: linear_probe "
        f"({trainable_params:,}/{total_params:,} trainable parameters)"
    )
    return model
