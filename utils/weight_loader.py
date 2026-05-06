import os

from peft import PeftModel
from safetensors.torch import load_file

from models.encoder_classifier import LongTableEncoderClassifier
from models.TableEncoder.encoder import LongTableEncoderMemory


def _checkpoint_file(pretrained_path: str) -> str:
    if not os.path.isdir(pretrained_path):
        raise NotADirectoryError(f"pretrained_path must be a checkpoint directory: {pretrained_path}")
    return os.path.join(pretrained_path, "model.safetensors")


def load_encoder_weights(encoder: LongTableEncoderMemory, pretrained_path: str):
    checkpoint_path = _checkpoint_file(pretrained_path)
    state_dict = load_file(checkpoint_path)
    encoder.load_state_dict(state_dict, strict=True)
    return encoder


def load_encoder_classifier_weights(model: LongTableEncoderClassifier, pretrained_path: str):
    checkpoint_path = _checkpoint_file(pretrained_path)
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    return model


def load_lora_weights(model: LongTableEncoderClassifier, pretrained_path: str, is_trainable: bool = False):
    if not os.path.isdir(pretrained_path):
        raise NotADirectoryError(f"pretrained_path must be a LoRA adapter directory: {pretrained_path}")
    model = PeftModel.from_pretrained(model, pretrained_path, is_trainable=is_trainable)
    return model


def load_model_weights(
    model: LongTableEncoderClassifier,
    pretrained_path: str = None,
    use_lora: bool = False,
    is_trainable: bool = False,
    checkpoint_type: str = "classifier",
):
    if not pretrained_path:
        return model

    if use_lora:
        checkpoint_type = "lora"

    if checkpoint_type == "encoder":
        if is_trainable:
            raise ValueError("is_trainable is only valid for LoRA checkpoints.")
        model.encoder = load_encoder_weights(model.encoder, pretrained_path)
        return model
    if checkpoint_type == "classifier":
        if is_trainable:
            raise ValueError("is_trainable is only valid for LoRA checkpoints.")
        return load_encoder_classifier_weights(model, pretrained_path)
    if checkpoint_type == "lora":
        return load_lora_weights(model, pretrained_path, is_trainable=is_trainable)

    raise ValueError("checkpoint_type must be one of: encoder, classifier, lora")
