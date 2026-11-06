"""Sentence embeddings for deduplication.

The encoder is pinned by model ID and revision and runs on CPU in fp32, with
the Hugging Face hub set to offline and telemetry off. Embeddings are saved
next to the build outputs with the texts they belong to, and reused when the
texts, the model, its revision and the library versions all match, so a
rebuild does not depend on recomputing them.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _text_digest(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def encode(
    texts: list[str], model_name: str, revision: str, cache_dir: Path
) -> dict[str, np.ndarray]:
    """Normalised embeddings for the distinct texts, read from or written to `cache_dir`."""
    distinct = sorted(set(texts))
    digest = _text_digest(distinct)
    vectors_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "embeddings.json"

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import sentence_transformers
    import torch

    key = (digest, model_name, revision, sentence_transformers.__version__, torch.__version__)
    if vectors_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cached = (
            meta["texts_sha256"],
            meta["model"],
            meta["revision"],
            meta["sentence_transformers"],
            meta["torch"],
        )
        if cached == key:
            vectors = np.load(vectors_path)
            return dict(zip(distinct, vectors))

    torch.manual_seed(0)
    model = sentence_transformers.SentenceTransformer(model_name, revision=revision, device="cpu")
    vectors = model.encode(
        distinct, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    meta = {
        "model": model_name,
        "revision": revision,
        "texts_sha256": digest,
        "count": len(distinct),
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return dict(zip(distinct, vectors))
