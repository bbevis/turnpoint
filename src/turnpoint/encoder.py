"""Frozen MPNet turn embeddings with persistent caching and turn metadata."""

import hashlib
from math import log1p
from pathlib import Path
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torch import nn

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


class TurnEncoder(nn.Module):
    """Encode each turn independently with frozen, revision-pinned MPNet.

    cache_dir stores text embeddings locally across runs; no outcomes or speaker
    metadata enter this cache. Long turns are split into consecutive token chunks
    within the model limit. Chunk embeddings are averaged by token count, then
    L2-normalised. Short turns are encoded directly. No text is silently truncated.
    encoding_device defaults to Apple MPS when available, otherwise CPU.
    Returned vectors follow this module's device (CPU by default).
    """

    def __init__(self, cache_dir=".cache/turnpoint", encoding_device=None):
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.encoding_device = encoding_device or ("mps" if torch.backends.mps.is_available() else "cpu")
        # Keep the frozen text model separate from trainable module state.
        object.__setattr__(self, "text_model", SentenceTransformer(
            MODEL_NAME, revision=MODEL_REVISION, device=self.encoding_device))
        self.text_model.eval()
        self.text_model.requires_grad_(False)
        self.output_dim = self.text_model.get_embedding_dimension() + 2
        tokenizer = self.text_model.tokenizer
        self.chunk_size = self.text_model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(cache_dir / "mpnet.sqlite3")
        self.cache.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    def _key(self, text):
        specification = f"{MODEL_NAME}:{MODEL_REVISION}:{self.chunk_size}:token-weighted-v1:{text}"
        return hashlib.sha256(specification.encode("utf-8")).hexdigest()

    def prepare(self, texts):
        """Cache missing texts in batches; safe to reuse across data splits.

        Frozen pretrained weights and deterministic per-text transformations do
        not fit anything to validation or test data. Persistent cache entries are
        keyed by exact text, model revision, and chunking policy.
        """
        texts = list(dict.fromkeys(texts))
        for start in range(0, len(texts), 512):
            pending = [text for text in texts[start:start + 512]
                       if self.cache.execute("SELECT 1 FROM embeddings WHERE key=?", (self._key(text),)).fetchone() is None]
            chunks, owners, weights = [], [], []
            for owner, text in enumerate(pending):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Turn text must be nonblank strings")
                tokens = self.text_model.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
                if len(tokens) <= self.chunk_size:
                    chunks.append(tokens); owners.append(owner); weights.append(len(tokens) or 1)
                else:
                    for offset in range(0, len(tokens), self.chunk_size):
                        part = tokens[offset:offset + self.chunk_size]
                        chunks.append(part)
                        owners.append(owner); weights.append(len(part))
            if not chunks:
                continue
            vectors = []
            tokenizer = self.text_model.tokenizer
            with torch.no_grad():
                for offset in range(0, len(chunks), 32):
                    inputs = [{"input_ids": [tokenizer.cls_token_id] + part + [tokenizer.sep_token_id],
                               "attention_mask": [1] * (len(part) + tokenizer.num_special_tokens_to_add(pair=False))}
                              for part in chunks[offset:offset + 32]]
                    features = tokenizer.pad(inputs, padding=True, return_tensors="pt")
                    features = {key: value.to(self.encoding_device) for key, value in features.items()}
                    embedded = self.text_model(features)["sentence_embedding"]
                    vectors.extend(torch.nn.functional.normalize(embedded, dim=1).cpu().numpy())
            pooled = np.zeros((len(pending), self.output_dim - 2), dtype=np.float32)
            for vector, owner, weight in zip(vectors, owners, weights):
                pooled[owner] += vector * weight
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            self.cache.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                                   [(self._key(text), vector.tobytes()) for text, vector in zip(pending, pooled)])
            self.cache.commit()
        print(f"Turn embeddings ready: {len(texts):,} unique texts.", flush=True)

    def forward(self, turns):
        """Return vectors: 768 text dimensions plus log word count/speaker change.

        Turn position and target-speaker information are added by ConversationModel.
        Missing embeddings are computed and cached. No outcome labels are read.
        """
        if not turns:
            raise ValueError("Provide at least one turn")
        texts = [turn["text"] for turn in turns]
        records = [self.cache.execute("SELECT vector FROM embeddings WHERE key=?", (self._key(text),)).fetchone() for text in texts]
        if any(record is None for record in records):
            self.prepare(texts)
            records = [self.cache.execute("SELECT vector FROM embeddings WHERE key=?", (self._key(text),)).fetchone() for text in texts]
        semantic = torch.tensor(np.stack([np.frombuffer(record[0], dtype=np.float32) for record in records]), device=self._device_anchor.device)
        metadata = semantic.new_tensor([
            [log1p(len(t["text"].split())), float(i > 0 and t["speaker_id"] != turns[i - 1]["speaker_id"])]
            for i, t in enumerate(turns)])
        return {"vectors": torch.cat([semantic, metadata], dim=1)}
