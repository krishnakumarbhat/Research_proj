from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import set_global_seed


@dataclass
class ToyLanguageModelResult:
    name: str
    val_nll: float
    activation_sparsity: float
    runtime_sec: float
    parameter_count: int


def build_character_corpus(texts: list[str], *, max_chars: int = 60_000) -> tuple[str, dict[str, int], dict[int, str]]:
    joined = "\n\n".join(text.strip() for text in texts if text and text.strip())[:max_chars]
    vocab = sorted(set(joined))
    char_to_id = {char: index for index, char in enumerate(vocab)}
    id_to_char = {index: char for char, index in char_to_id.items()}
    return joined, char_to_id, id_to_char


def make_next_token_datasets(
    texts: list[str],
    *,
    context_size: int = 24,
    max_train_samples: int = 5000,
    max_val_samples: int = 1200,
) -> tuple[TensorDataset, TensorDataset, list[int], list[int], int]:
    joined, char_to_id, _ = build_character_corpus(texts)
    token_ids = [char_to_id[char] for char in joined]

    split = max(context_size + 2, int(len(token_ids) * 0.82))
    train_ids = token_ids[:split]
    val_ids = token_ids[split - context_size :]

    def build_examples(ids: list[int], max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        span = max(0, len(ids) - context_size - 1)
        if span == 0:
            empty_x = torch.empty((0, context_size), dtype=torch.long)
            empty_y = torch.empty((0,), dtype=torch.long)
            return empty_x, empty_y
        stride = max(1, span // max_samples)
        contexts = []
        targets = []
        for start in range(0, span, stride):
            contexts.append(ids[start : start + context_size])
            targets.append(ids[start + context_size])
            if len(contexts) >= max_samples:
                break
        return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    train_x, train_y = build_examples(train_ids, max_train_samples)
    val_x, val_y = build_examples(val_ids, max_val_samples)
    return (
        TensorDataset(train_x, train_y),
        TensorDataset(val_x, val_y),
        train_ids,
        val_ids,
        len(char_to_id),
    )


def bigram_baseline(train_ids: list[int], val_ids: list[int], vocab_size: int) -> ToyLanguageModelResult:
    start = time.perf_counter()
    counts = np.ones((vocab_size, vocab_size), dtype=np.float64)
    for left, right in zip(train_ids[:-1], train_ids[1:]):
        counts[left, right] += 1.0
    probabilities = counts / counts.sum(axis=1, keepdims=True)

    losses = []
    for left, right in zip(val_ids[:-1], val_ids[1:]):
        losses.append(-np.log(probabilities[left, right]))

    runtime = time.perf_counter() - start
    return ToyLanguageModelResult(
        name="bigram",
        val_nll=float(np.mean(losses)) if losses else float("inf"),
        activation_sparsity=0.0,
        runtime_sec=runtime,
        parameter_count=int(vocab_size * vocab_size),
    )


class TinyMLPLM(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, d_model: int = 24, hidden_dim: int = 96) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(context_size * d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(inputs)
        hidden = F.gelu(self.fc1(embedded.reshape(inputs.size(0), -1)))
        logits = self.fc2(hidden)
        if return_hidden:
            return logits, hidden
        return logits


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(inputs, inputs, inputs, need_weights=True)
        hidden = self.ln1(inputs + attn_out)
        ff_out = self.ff(hidden)
        return self.ln2(hidden + ff_out), attn_weights


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, d_model: int = 48, nhead: int = 4, ff_dim: int = 96) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context_size, d_model)
        self.block = TinyTransformerBlock(d_model, nhead, ff_dim)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0)
        hidden = self.embedding(inputs) + self.position(positions)
        hidden, attn_weights = self.block(hidden)
        final_hidden = hidden[:, -1, :]
        logits = self.head(final_hidden)
        if return_hidden:
            return logits, final_hidden, attn_weights
        return logits


def _evaluate_model(model: nn.Module, data_loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    losses = []
    activations = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            if isinstance(model, TinyMLPLM):
                logits, hidden = model(features, return_hidden=True)
                activations.append(hidden.detach().abs())
            else:
                logits, hidden, _ = model(features, return_hidden=True)
                activations.append(hidden.detach().abs())
            losses.append(F.cross_entropy(logits, labels).item())
    sparsity = 0.0
    if activations:
        stacked = torch.cat(activations)
        sparsity = float((stacked < 1e-3).float().mean().item())
    return float(np.mean(losses)) if losses else float("inf"), sparsity


def _train_model(
    model: nn.Module,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    *,
    device: str,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 3e-3,
) -> ToyLanguageModelResult:
    start = time.perf_counter()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

    val_nll, sparsity = _evaluate_model(model, val_loader, device)
    runtime = time.perf_counter() - start
    return ToyLanguageModelResult(
        name=model.__class__.__name__.replace("Tiny", "").replace("LM", "").lower(),
        val_nll=val_nll,
        activation_sparsity=sparsity,
        runtime_sec=runtime,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def run_toy_language_model_benchmark(texts: list[str], *, seed: int = 42) -> list[ToyLanguageModelResult]:
    set_global_seed(seed)
    train_dataset, val_dataset, train_ids, val_ids, vocab_size = make_next_token_datasets(texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Not enough text to train the toy language models.")

    context_size = train_dataset.tensors[0].size(1)
    results = [bigram_baseline(train_ids, val_ids, vocab_size)]
    results.append(
        _train_model(
            TinyMLPLM(vocab_size=vocab_size, context_size=context_size),
            train_dataset,
            val_dataset,
            device=device,
        )
    )
    results.append(
        _train_model(
            TinyTransformerLM(vocab_size=vocab_size, context_size=context_size),
            train_dataset,
            val_dataset,
            device=device,
        )
    )
    return results