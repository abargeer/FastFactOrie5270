"""Training loops for LoRA SFT, frozen-encoder RAG baseline, and DPO-style tuning.

The three loops are deliberately kept separate (instead of a single
parameterized one) so each remains easy to read on its own. They share most
of their structure: AdamW + cosine schedule, gradient accumulation, optional
mixed-precision, AUC-based early-best selection on a validation set.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fast_fact.config import ModelConfig
from fast_fact.data.datasets import collate_fn
from fast_fact.models.base import apply_lora, build_base_model, freeze_encoder
from fast_fact.models.evaluate import evaluate_classifier


def _make_loaders(train_ds: Dataset, val_ds: Dataset, batch_size: int):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


def _make_optimizer(params, model_cfg: ModelConfig):
    return torch.optim.AdamW(
        params, lr=model_cfg.lr, weight_decay=model_cfg.weight_decay
    )


def _make_scheduler(optimizer, model_cfg: ModelConfig, num_training_steps: int):
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=model_cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_supervised_lora(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer,
    model_cfg: ModelConfig,
    output_dir: str,
    device: Optional[torch.device] = None,
):
    """LoRA-SFT classifier training with AUC-based best-epoch checkpointing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_base_model(model_cfg, num_labels=2)
    model = apply_lora(model, model_cfg)
    model.to(device)

    train_loader, val_loader = _make_loaders(train_ds, val_ds, model_cfg.batch_size)
    optimizer = _make_optimizer(model.parameters(), model_cfg)
    num_training_steps = model_cfg.num_epochs * math.ceil(
        len(train_loader) / model_cfg.grad_accum_steps
    )
    scheduler = _make_scheduler(optimizer, model_cfg, num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_auc = -1.0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(model_cfg.num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / model_cfg.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += loss.item() * model_cfg.grad_accum_steps
            if (step + 1) % model_cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1}: train loss {avg_loss:.4f}")
        metrics = evaluate_classifier(model, val_loader, device)
        print(f"Epoch {epoch+1}: val metrics {metrics}")

        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            save_path = os.path.join(output_dir, "best_lora.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best LoRA model to {save_path}")

    best_path = os.path.join(output_dir, "best_lora.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def train_rag_baseline(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer,
    model_cfg: ModelConfig,
    output_dir: str,
    device: Optional[torch.device] = None,
):
    """RAG baseline: frozen encoder, retrained head, RAG-augmented input text."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_base_model(model_cfg, num_labels=2)
    model = freeze_encoder(model)
    model.to(device)

    train_loader, val_loader = _make_loaders(train_ds, val_ds, model_cfg.batch_size)
    optimizer = _make_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()), model_cfg
    )
    num_training_steps = model_cfg.num_epochs * math.ceil(
        len(train_loader) / model_cfg.grad_accum_steps
    )
    scheduler = _make_scheduler(optimizer, model_cfg, num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_auc = -1.0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(model_cfg.num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"RAG Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / model_cfg.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += loss.item() * model_cfg.grad_accum_steps
            if (step + 1) % model_cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"RAG Epoch {epoch+1}: train loss {avg_loss:.4f}")
        metrics = evaluate_classifier(model, val_loader, device)
        print(f"RAG Epoch {epoch+1}: val metrics {metrics}")

        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            save_path = os.path.join(output_dir, "best_rag.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best RAG model to {save_path}")

    best_path = os.path.join(output_dir, "best_rag.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model


def train_dpo(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer,
    model_cfg: ModelConfig,
    lora_sft_model,
    output_dir: str,
    device: Optional[torch.device] = None,
):
    """DPO-style preference tuning on top of an already-trained LoRA SFT model.

    The same LoRA-SFT model is deep-copied twice: once as the trainable policy
    and once as a frozen reference. The DPO objective compares
    log p(y|x) - log p(1-y|x) between policy and reference for each example.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = copy.deepcopy(lora_sft_model).to(device)
    reference = copy.deepcopy(lora_sft_model).to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    train_loader, val_loader = _make_loaders(train_ds, val_ds, model_cfg.batch_size)
    optimizer = _make_optimizer(policy.parameters(), model_cfg)
    num_training_steps = model_cfg.num_epochs * math.ceil(
        len(train_loader) / model_cfg.grad_accum_steps
    )
    scheduler = _make_scheduler(optimizer, model_cfg, num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_auc = -1.0
    os.makedirs(output_dir, exist_ok=True)
    beta = model_cfg.dpo_beta

    for epoch in range(model_cfg.num_epochs):
        policy.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"DPO Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logp_pol = torch.log_softmax(
                    policy(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits,
                    dim=-1,
                )
                lp_ch_pol = logp_pol.gather(
                    1, batch["chosen_labels"].unsqueeze(1)
                ).squeeze(1)
                lp_rj_pol = logp_pol.gather(
                    1, batch["rejected_labels"].unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():
                    logp_ref = torch.log_softmax(
                        reference(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits,
                        dim=-1,
                    )
                    lp_ch_ref = logp_ref.gather(
                        1, batch["chosen_labels"].unsqueeze(1)
                    ).squeeze(1)
                    lp_rj_ref = logp_ref.gather(
                        1, batch["rejected_labels"].unsqueeze(1)
                    ).squeeze(1)

                logit = beta * ((lp_ch_pol - lp_rj_pol) - (lp_ch_ref - lp_rj_ref))
                loss = -torch.nn.functional.logsigmoid(logit).mean()
                loss = loss / model_cfg.grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += loss.item() * model_cfg.grad_accum_steps
            if (step + 1) % model_cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"DPO Epoch {epoch+1}: train loss {avg_loss:.4f}")
        metrics = evaluate_classifier(policy, val_loader, device)
        print(f"DPO Epoch {epoch+1}: val metrics {metrics}")

        if metrics["auc"] > best_val_auc:
            best_val_auc = metrics["auc"]
            save_path = os.path.join(output_dir, "best_dpo.pt")
            torch.save(policy.state_dict(), save_path)
            print(f"Saved best DPO model to {save_path}")

    best_path = os.path.join(output_dir, "best_dpo.pt")
    if os.path.exists(best_path):
        policy.load_state_dict(torch.load(best_path, map_location=device))
    return policy
