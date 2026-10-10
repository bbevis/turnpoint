"""Trainable turn representations built with PyTorch embeddings, GRU, and SAE."""

from collections import Counter
from math import log1p
import re

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def _tokens(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def build_vocabulary(texts, min_count=2, max_words=20000):
    """Build a token-to-index dictionary from training turn texts only.

    Lowercases text and separates words and punctuation. Keeps at most
    max_words tokens occurring at least min_count times, plus padding (0)
    and unknown (1). Pass an iterable of strings; do not include validation
    or test conversations. Reuse the returned dictionary for all splits.
    """
    if min_count < 1 or max_words < 1:
        raise ValueError("min_count and max_words must be positive")
    counts = Counter(token for text in texts for token in _tokens(text))
    words = sorted(counts, key=lambda word: (-counts[word], word))
    words = [word for word in words if counts[word] >= min_count][:max_words]
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


class TurnEncoder(nn.Module):
    """Encode ordered turns using learned embeddings, a GRU, and sparse features.

    Parameters
    ----------
    vocabulary : dict
        Dictionary returned by build_vocabulary; keep it with the model weights.
    embedding_dim, hidden_dim, sparse_dim : int
        Word embedding, GRU state, and sparse feature sizes (100, 128, 512).

    Weights start untrained. This component is intended for joint training with
    the conversation outcome model.
    """

    def __init__(self, vocabulary, embedding_dim=100, hidden_dim=128, sparse_dim=512):
        super().__init__()
        if vocabulary.get("<pad>") != 0 or vocabulary.get("<unk>") != 1:
            raise ValueError("Use a vocabulary from build_vocabulary")
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("Vocabulary indices must be contiguous")
        self.vocabulary = dict(vocabulary)
        self.embedding = nn.Embedding(len(vocabulary), embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.sparse_encoder = nn.Linear(hidden_dim, sparse_dim)
        self.decoder = nn.Linear(sparse_dim, hidden_dim)

    def forward(self, turns):
        """Encode one conversation's ordered turn dictionaries from the loader.

        Inputs require text and speaker_id. Returns a dictionary containing:
        vectors: [number of turns, sparse_dim + 2], sparse features followed by
        log(1 + whitespace word count) and a speaker-change indicator (first=0).
        semantic_features: [number of turns, sparse_dim] ReLU activations.
        reconstruction_loss: mean squared error reconstructing GRU states.
        sparsity_loss: mean absolute feature activation, before any weighting.

        Speaker changes use equality, so arbitrary speaker relabelling leaves
        vectors unchanged. This describes dyadic speaker structure, not named
        roles or full identity in multi-party interactions. Outcome ownership
        remains attached to the original IDs in the loader's output.
        Pass a complete conversation, not mixed or reordered conversation turns.
        Loss weights and the outcome objective are handled by train_model.
        """
        if not turns:
            raise ValueError("Provide at least one turn")
        if any(not isinstance(t["text"], str) or not t["text"].strip() for t in turns):
            raise ValueError("Turn text must be nonblank strings")
        device = self.embedding.weight.device
        tokens = [torch.tensor([self.vocabulary.get(word, 1) for word in _tokens(t["text"])],
                               dtype=torch.long, device=device) for t in turns]
        lengths = torch.tensor([len(sequence) for sequence in tokens], device="cpu")
        embedded = self.embedding(pad_sequence(tokens, batch_first=True))
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, state = self.gru(packed)
        dense = state[-1]
        semantic = F.relu(self.sparse_encoder(dense))
        # Unit decoder columns prevent reducing the L1 loss by rescaling weights.
        reconstruction = F.linear(semantic, F.normalize(self.decoder.weight, dim=0), self.decoder.bias)
        metadata = dense.new_tensor([
            [log1p(len(t["text"].split())), float(i > 0 and t["speaker_id"] != turns[i - 1]["speaker_id"])]
            for i, t in enumerate(turns)
        ])
        return {"vectors": torch.cat([semantic, metadata], dim=1),
                "semantic_features": semantic,
                "reconstruction_loss": F.mse_loss(reconstruction, dense),
                "sparsity_loss": semantic.abs().mean()}
