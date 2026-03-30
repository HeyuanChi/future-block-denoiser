from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class DataConfig:
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer_name: str = "bert-base-uncased"
    task_mode: str = "future"
    train_split: str = "train"
    val_split: str = "validation"
    source_field: str = "text"
    target_field: str = "text"
    prefix_len: int = 64
    future_len: int = 16
    suffix_len: int = 0
    stride: int = 16
    batch_size: int = 8
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DataConfig":
        data_config = config.get("data", config)
        return cls(**data_config)

    @property
    def total_len(self) -> int:
        return self.prefix_len + self.future_len + self.suffix_len


class PrefixFutureDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of fixed windows split into prefix and future blocks."""

    def __init__(
        self,
        token_ids: list[int],
        prefix_len: int,
        future_len: int,
        suffix_len: int,
        task_mode: str,
        stride: int,
        max_samples: int | None = None,
    ) -> None:
        self.task_mode = task_mode
        self.prefix_len = prefix_len
        self.future_len = future_len
        self.suffix_len = suffix_len
        self.total_len = prefix_len + future_len + suffix_len
        self.samples = self._build_samples(
            token_ids=token_ids,
            stride=stride,
            max_samples=max_samples,
        )

    def _build_samples(
        self,
        token_ids: list[int],
        stride: int,
        max_samples: int | None,
    ) -> list[dict[str, torch.Tensor]]:
        samples: list[dict[str, torch.Tensor]] = []

        max_start = len(token_ids) - self.total_len + 1
        if max_start <= 0:
            return samples

        for start in range(0, max_start, stride):
            window = token_ids[start : start + self.total_len]
            prefix_ids = torch.tensor(window[: self.prefix_len], dtype=torch.long)
            future_ids = torch.tensor(window[self.prefix_len : self.prefix_len + self.future_len], dtype=torch.long)

            sample = {
                "prefix_ids": prefix_ids,
                "future_ids": future_ids,
                "prefix_mask": torch.ones(self.prefix_len, dtype=torch.long),
                "future_mask": torch.ones(self.future_len, dtype=torch.long),
            }
            if self.task_mode == "infilling":
                suffix_ids = torch.tensor(window[self.prefix_len + self.future_len :], dtype=torch.long)
                sample["suffix_ids"] = suffix_ids
                sample["suffix_mask"] = torch.ones(self.suffix_len, dtype=torch.long)

            samples.append(sample)

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class Seq2SeqDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of fixed source-target pairs for seq2seq training."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        source_field: str,
        target_field: str,
        source_len: int,
        target_len: int,
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None:
            rows = rows[:max_samples]
        self.samples = self._build_samples(
            rows=rows,
            tokenizer=tokenizer,
            source_field=source_field,
            target_field=target_field,
            source_len=source_len,
            target_len=target_len,
        )

    def _build_samples(
        self,
        rows: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        source_field: str,
        target_field: str,
        source_len: int,
        target_len: int,
    ) -> list[dict[str, torch.Tensor]]:
        samples: list[dict[str, torch.Tensor]] = []
        for row in rows:
            source_text = str(row[source_field]).strip()
            target_text = str(row[target_field]).strip()
            if not source_text or not target_text:
                continue

            source_encoding = tokenizer(
                source_text,
                padding="max_length",
                truncation=True,
                max_length=source_len,
                return_tensors="pt",
            )
            target_encoding = tokenizer(
                target_text,
                padding="max_length",
                truncation=True,
                max_length=target_len,
                return_tensors="pt",
            )

            samples.append(
                {
                    "prefix_ids": source_encoding["input_ids"][0].long(),
                    "prefix_mask": source_encoding["attention_mask"][0].long(),
                    "future_ids": target_encoding["input_ids"][0].long(),
                    "future_mask": target_encoding["attention_mask"][0].long(),
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def build_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # We tokenize a very long concatenated corpus and then slice fixed windows ourselves.
    # Setting a large max length avoids irrelevant warnings from the tokenizer.
    tokenizer.model_max_length = 10**9
    return tokenizer


def load_split_texts(dataset_name: str, dataset_config: str, split: str) -> list[str]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    return [row["text"] for row in dataset if row["text"].strip()]


def load_split_rows(dataset_name: str, dataset_config: str, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    return [dict(row) for row in dataset]


def tokenize_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    merged_text = "\n\n".join(texts)
    encoded = tokenizer(
        merged_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return list(encoded["input_ids"])


def build_dataset(
    config: DataConfig,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset[dict[str, torch.Tensor]]:
    max_samples = config.max_train_samples if split == "train" else config.max_val_samples
    if config.task_mode == "seq2seq":
        rows = load_split_rows(config.dataset_name, config.dataset_config, split)
        return Seq2SeqDataset(
            rows=rows,
            tokenizer=tokenizer,
            source_field=config.source_field,
            target_field=config.target_field,
            source_len=config.prefix_len,
            target_len=config.future_len,
            max_samples=max_samples,
        )

    texts = load_split_texts(config.dataset_name, config.dataset_config, split)
    token_ids = tokenize_texts(texts, tokenizer)
    return PrefixFutureDataset(
        token_ids=token_ids,
        prefix_len=config.prefix_len,
        future_len=config.future_len,
        suffix_len=config.suffix_len,
        task_mode=config.task_mode,
        stride=config.stride,
        max_samples=max_samples,
    )


def build_dataloaders(
    config: DataConfig,
) -> tuple[PreTrainedTokenizerBase, DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]]]:
    tokenizer = build_tokenizer(config.tokenizer_name)
    train_dataset = build_dataset(config, split=config.train_split, tokenizer=tokenizer)
    val_dataset = build_dataset(config, split=config.val_split, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return tokenizer, train_loader, val_loader
