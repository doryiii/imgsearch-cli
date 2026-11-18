"""Shared types and helpers for remote and local embedding backends."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import numpy as np

QUERY_PREFIX = "task: search result | query: "
DEFAULT_HF_MODEL = "google/embeddinggemma-2"
CANONICAL_GEMMA2_ID = "embeddinggemma-2"

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
  """Connection error, timeout, 5xx, or systemic server failure."""


class ModelMismatch(EmbedError):
  """Backend model does not match the configured or indexed model."""


class ItemRejected(EmbedError):
  """Backend rejected a specific payload item."""


class EmbeddingBackend(Protocol):
  """Interface implemented by remote HTTP and local CPU/GPU backends."""

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
  """Detect image MIME type from magic bytes with extension fallback."""
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
  """Defensive L2 normalization to float32, rejecting NaN/Inf values."""
  arr = np.asarray(vector, dtype=np.float32)
  if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
    raise EmbedError("invalid embedding vector (empty, NaN, or Inf)")
  norm = float(np.linalg.norm(arr))
  return arr / norm if norm > 0 else arr


def is_embeddinggemma2(model_id: str | None) -> bool:
  """Return True if `model_id` names EmbeddingGemma 2 (HF or GGUF)."""
  if not model_id:
    return False
  low = model_id.lower()
  tokens = ("embeddinggemma-2", "embeddinggemma2", "embedding_gemma2")
  return any(tok in low for tok in tokens)


def canonical_model_id(model_id: str | None) -> str | None:
  """Normalize EmbeddingGemma 2 aliases to a single canonical model key."""
  if not model_id:
    return None
  stripped = model_id.strip()
  if not stripped:
    return None
  if is_embeddinggemma2(stripped):
    return CANONICAL_GEMMA2_ID
  return stripped


def models_compatible(a: str | None, b: str | None) -> bool:
  """Return True when two model identifiers share the same vector space."""
  if not a or not b:
    return False
  if is_embeddinggemma2(a) or is_embeddinggemma2(b):
    return is_embeddinggemma2(a) and is_embeddinggemma2(b)
  la = a.strip().lower()
  lb = b.strip().lower()
  if not la or not lb:
    return False
  if la == lb:
    return True
  base_a = la.rsplit("/", 1)[-1]
  base_b = lb.rsplit("/", 1)[-1]
  return bool(base_a) and base_a == base_b


def resolve_local_hf_model(model_id: str | None) -> str:
  """Map a CLI/DB model identifier to an HF repo ID or local directory."""
  if not model_id or not model_id.strip():
    return DEFAULT_HF_MODEL
  cleaned = model_id.strip()
  if Path(cleaned).is_dir():
    return cleaned
  low = cleaned.lower()
  if is_embeddinggemma2(cleaned):
    if "/" not in cleaned or "gguf" in low:
      return DEFAULT_HF_MODEL
    return cleaned
  return cleaned


def decode_rgb_image(source: bytes | Path):
  """Decode image bytes or path to RGB, compositing alpha onto dark gray."""
  try:
    from PIL import Image, ImageOps
  except ImportError as exc:
    raise EmbedError(
        "Pillow is required for image decoding (`pip install pillow`)"
    ) from exc

  stream = io.BytesIO(source) if isinstance(source, bytes) else source
  try:
    with Image.open(stream) as im:
      im = ImageOps.exif_transpose(im)
      has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
      if has_alpha:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (30, 30, 30, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
      return im.convert("RGB")
  except Image.DecompressionBombError as exc:
    raise ValueError(f"image exceeds decompression limit: {exc}") from exc
