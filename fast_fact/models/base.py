"""Build a sequence classifier, attach LoRA adapters, or freeze the encoder."""

from __future__ import annotations

from fast_fact.config import ModelConfig


def build_base_model(model_cfg: ModelConfig, num_labels: int = 2):
    """Instantiate a Hugging Face sequence classification model."""
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_cfg.base_model,
        num_labels=num_labels,
    )


def apply_lora(model, model_cfg: ModelConfig):
    """Wrap ``model`` with PEFT LoRA adapters on attention projections."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query", "key", "value"],
    )
    return get_peft_model(model, lora_cfg)


def freeze_encoder(model):
    """Freeze the backbone encoder, leaving only the classification head trainable.

    Walks a small list of common attribute names for HF backbones; raises if
    none of them are found.
    """
    base_attr = None
    for attr in ["bert", "roberta", "deberta", "base_model"]:
        if hasattr(model, attr):
            base_attr = attr
            break
    if base_attr is None:
        raise ValueError("Could not find encoder attribute on model.")
    encoder = getattr(model, base_attr)
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Encoder '{base_attr}' frozen; only classification head trainable.")
    return model
