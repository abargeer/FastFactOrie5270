import sys
import types
from unittest.mock import MagicMock

import pytest

from fast_fact.config import ModelConfig
from fast_fact.models import base as base_mod


def test_freeze_encoder_freezes_roberta(tiny_classifier_factory):
    model = tiny_classifier_factory()
    base_mod.freeze_encoder(model)
    assert all(not p.requires_grad for p in model.roberta.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_freeze_encoder_raises_when_no_known_attribute():
    class NoEncoder:
        pass
    with pytest.raises(ValueError):
        base_mod.freeze_encoder(NoEncoder())


def test_build_base_model_calls_transformers(monkeypatch):
    fake_model = MagicMock(name="hf_model")

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, num_labels):
            return fake_model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForSequenceClassification = FakeAuto
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    out = base_mod.build_base_model(ModelConfig(base_model="dummy"), num_labels=2)
    assert out is fake_model


def test_apply_lora_calls_peft(monkeypatch):
    captured = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_get_peft_model(model, cfg):
        captured["called"] = True
        return ("wrapped", model)

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.get_peft_model = fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    out = base_mod.apply_lora("model", ModelConfig())
    assert out == ("wrapped", "model")
    assert captured["called"] is True
    assert captured["task_type"] == "SEQ_CLS"
