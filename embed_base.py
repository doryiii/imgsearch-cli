"""Shared types, constants, and utilities for remote and local embedding backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

QUERY_PREFIX = "task: search result | query: "
DEFAULT_HF_MODEL = "google/embeddinggemma-2"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
EXT_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


class EmbedError(Exception):
  """Base error for embedding backend failures."""


class EndpointUnreachable(EmbedError):
  """Connection error, timeout, 5xx, or malformed response."""


class ModelMismatch(EmbedError):
  """Backend model does not match the configured or indexed model."""


class ItemRejected(EmbedError):
  """Backend rejected a specific payload item."""


class EmbeddingBackend(Protocol):
  """Common interface implemented by both remote HTTP and local CPU/GPU backends."""

  model_id: str | None
  served_model: str | None

  def close(self) -> None:
    ...

  def probe(self) -> dict:
    ...

  def verify(self) -> str:
    ...

  def embed_images(self, items: list[tuple[bytes, str]]) -> list[np.ndarray]:
    ...

  def embed_text(self, query: str) -> np.ndarray:
    ...


def detect_mime(data: bytes, path: Path | None = None) -> str:
  """Detect image MIME type from magic bytes with file-extension fallback."""
  if data.startswith(b"\x89PNG\r\n\x1a\n"):
    return "image/png"
  if data.startswith(b"\xff\xd8\xff"):
    return "image/jpeg"
  if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
    return "image/webp"
  if data.startswith((b"GIF87a", b"GIF89a")):
    return "image/gif"
  if data.startswith(b"BM"):
    return "image/bmp"
  if path is not None:
    return EXT_MIMES.get(path.suffix.lower(), "image/png")
  return "image/png"


def normalize(vector) -> np.ndarray:
  """Defensive L2 normalization to float32."""
  arr = np.asarray(vector, dtype=np.float32)
  norm = float(np.linalg.norm(arr))
  return arr / norm if norm > 0 else arr


def is_embeddinggemma2(model_id: str | None) -> bool:
  """Return True if `model_id` refers to EmbeddingGemma 2 under HF or GGUF/llama.cpp naming."""
  if not model_id:
    return False
  low = model_id.lower()
  return any(tok in low for tok in ("embeddinggemma-2", "embeddinggemma2", "embedding_gemma2"))


def models_compatible(a: str | None, b: str | None) -> bool:
  """Return True when two model identifiers share the same embedding vector space."""
  if not a or not b:
    return False
  la, lb = a.lower(), b.lower()
  if la == lb or la in lb or lb in la:
    return True
  return is_embeddinggemma2(a) and is_embeddinggemma2(b)


def resolve_local_hf_model(model_id: str | None) -> str:
  """Map a CLI/DB model identifier to a Hugging Face repo ID or local directory path."""
  if not model_id:
    return DEFAULT_HF_MODEL
  if Path(model_id).exists() or "/" in model_id:
    return model_id
  if is_embeddinggemma2(model_id):
    return DEFAULT_HF_MODEL
  return model_id
