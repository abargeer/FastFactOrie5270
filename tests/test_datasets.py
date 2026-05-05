import torch

from fast_fact.data.datasets import DPODataset, TextClassificationDataset, collate_fn


def test_text_classification_dataset_shapes(labeled_events_df, tiny_tokenizer):
    ds = TextClassificationDataset(labeled_events_df, tiny_tokenizer, "text", max_length=8)
    assert len(ds) == len(labeled_events_df)
    item = ds[0]
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"].dtype == torch.long


def test_dpo_dataset_chosen_and_rejected(labeled_events_df, tiny_tokenizer):
    ds = DPODataset(labeled_events_df, tiny_tokenizer, "text", max_length=8)
    item = ds[0]
    y = int(labeled_events_df.iloc[0]["label"])
    assert int(item["chosen_labels"]) == y
    assert int(item["rejected_labels"]) == 1 - y


def test_collate_fn_stacks(labeled_events_df, tiny_tokenizer):
    ds = TextClassificationDataset(labeled_events_df, tiny_tokenizer, "text", max_length=8)
    batch = collate_fn([ds[0], ds[1], ds[2]])
    assert batch["input_ids"].shape == (3, 8)
    assert batch["labels"].shape == (3,)
