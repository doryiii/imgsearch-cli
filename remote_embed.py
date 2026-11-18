"""Remote HTTP backend for llama.cpp / OpenAI `/v1/embeddings` servers."""

from __future__ import annotations

import base64
import io
import os

import httpx
import numpy as np

from embed_base import (
    EmbedError,
    EndpointUnreachable,
    ItemRejected,
    ModelMismatch,
    QUERY_PREFIX,
    canonical_model_id,
    decode_rgb_image,
    models_compatible,
    normalize,
)

DEFAULT_ENDPOINT = os.environ.get("EMBED_ENDPOINT") or None
DEFAULT_MODEL = os.environ.get("EMBED_MODEL") or None

EMBED_PATH = "/v1/embeddings"
MODELS_PATH = "/v1/models"
PROBE_TIMEOUT = 5.0
REQUEST_TIMEOUT = 300.0

EMBED_NAME_TOKENS = (
    "embed",
    "bge-",
    "bge_",
    "e5-",
    "gte-",
    "siglip",
    "jina-clip",
)
SYSTEMIC_HTTP_CODES = {401, 403, 404, 405, 408, 429, 501}
# llama.cpp returns HTTP 500 (not 400) when stb_image/mtmd fails on a bad file.
ITEM_REJECTED_HINTS = (
    "failed to load image",
    "failed to decode",
    "failed to preprocess",
    "invalid image",
    "is too large to process",
)
TRANSCODE_MAX_DIM = 2048


def _transcode_to_png(raw: bytes) -> bytes | None:
  """Try decoding `raw` with Pillow into a clean 8-bit RGB PNG."""
  try:
    from PIL import Image

    rgb = decode_rgb_image(raw)
    rgb.thumbnail(
        (TRANSCODE_MAX_DIM, TRANSCODE_MAX_DIM), Image.Resampling.LANCZOS
    )
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()
  except (OSError, ValueError, SyntaxError, EOFError, EmbedError):
    return None


def _as_str_list(val) -> list[str]:
  if isinstance(val, list):
    return [str(v) for v in val if v]
  if isinstance(val, str) and val:
    return [val]
  if isinstance(val, dict):
    return [str(k) for k, enabled in val.items() if k and enabled]
  return []


def _parse_models(payload: object) -> list[dict]:
  """Normalize OpenAI `data[]` and llama.cpp/Ollama `models[]` entries."""
  if not isinstance(payload, dict):
    raise ValueError("expected JSON object from /v1/models")
  raw_data = (
      payload.get("data") if isinstance(payload.get("data"), list) else []
  )
  raw_models = (
      payload.get("models") if isinstance(payload.get("models"), list) else []
  )

  models_by_id: dict[str, dict] = {}
  for m in raw_models:
    if isinstance(m, dict):
      mid = m.get("model") or m.get("name") or m.get("id")
      if mid:
        models_by_id[str(mid)] = m

  parsed: list[dict] = []
  source = raw_data if raw_data else raw_models
  for idx, item in enumerate(source):
    if not isinstance(item, dict):
      continue
    model_id = item.get("id") or item.get("model") or item.get("name")
    if not model_id:
      continue
    model_id = str(model_id)
    companion = models_by_id.get(model_id) or (
        raw_models[idx]
        if raw_data
        and idx < len(raw_models)
        and isinstance(raw_models[idx], dict)
        else {}
    )
    arch = (
        item.get("architecture")
        if isinstance(item.get("architecture"), dict)
        else {}
    )
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    details = (
        companion.get("details")
        if isinstance(companion.get("details"), dict)
        else {}
    )
    caps = _as_str_list(companion.get("capabilities")) or _as_str_list(
        item.get("capabilities")
    )
    tags = _as_str_list(item.get("tags")) + _as_str_list(companion.get("tags"))
    aliases = _as_str_list(item.get("aliases")) or _as_str_list(
        companion.get("aliases")
    )
    args = _as_str_list(status.get("args"))

    parsed.append(
        {
            "id": model_id,
            "aliases": aliases,
            "capabilities": [c.lower() for c in caps],
            "type": str(
                companion.get("type") or item.get("type") or ""
            ).lower(),
            "tags": [t.lower() for t in tags],
            "output_modalities": [
                m.lower() for m in _as_str_list(arch.get("output_modalities"))
            ],
            "status_args": [a.lower() for a in args],
            "family": str(details.get("family") or "").lower(),
        }
    )
  return parsed


def _is_embedding_model(entry: dict) -> bool:
  """Return True if a `/v1/models` entry looks like an embedding model."""
  if any("embed" in c for c in entry["capabilities"]):
    return True
  if "embed" in entry["type"]:
    return True
  if any("embed" in m for m in entry["output_modalities"]):
    return True
  if any("embed" in t for t in entry["tags"]):
    return True
  if any(a.startswith("--embedding") for a in entry["status_args"]):
    return True

  names = [entry["id"], *entry["aliases"], entry["family"]]
  for name in names:
    low = name.lower()
    if any(tok in low for tok in EMBED_NAME_TOKENS):
      return True
  return False


def _match_requested_model(needle: str, entries: list[dict]) -> str | None:
  """Match `needle` against `/v1/models`, preferring exact & embed models."""
  low = needle.strip().lower()
  for e in entries:
    if e["id"].lower() == low or any(a.lower() == low for a in e["aliases"]):
      return e["id"]
  for e in entries:
    if models_compatible(needle, e["id"]) or any(
        models_compatible(needle, a) for a in e["aliases"]
    ):
      return e["id"]
  for e in entries:
    if _is_embedding_model(e) and (
        low in e["id"].lower() or any(low in a.lower() for a in e["aliases"])
    ):
      return e["id"]
  for e in entries:
    if low in e["id"].lower() or any(low in a.lower() for a in e["aliases"]):
      return e["id"]
  return None


class EmbeddingClient:
  """Client for llama.cpp's multimodal `/v1/embeddings` endpoint."""

  def __init__(
      self,
      base_url: str,
      model_id: str | None = DEFAULT_MODEL,
      transport=None,
  ):
    if not base_url:
      raise ValueError("base_url is required for remote EmbeddingClient")
    self.base_url = base_url.rstrip("/")
    self.model_id = model_id
    self.served_model: str | None = None
    self._client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=PROBE_TIMEOUT),
    )
    self._verified = False

  def close(self) -> None:
    self._client.close()

  def __enter__(self) -> EmbeddingClient:
    return self

  def __exit__(self, *exc) -> None:
    self.close()

  def probe(self) -> dict:
    """Query GET /v1/models for reachability and resolve target model."""
    try:
      resp = self._client.get(
          self.base_url + MODELS_PATH, timeout=PROBE_TIMEOUT
      )
      resp.raise_for_status()
      entries = _parse_models(resp.json())
      available = [e["id"] for e in entries]
      embed_models = [e["id"] for e in entries if _is_embedding_model(e)]

      if self.model_id:
        matched = _match_requested_model(self.model_id, entries)
        return {
            "reachable": True,
            "served_model": matched,
            "available_models": available,
            "embedding_models": embed_models,
            "matches": matched is not None,
        }

      selected = embed_models[0] if embed_models else None
      return {
          "reachable": True,
          "served_model": selected,
          "available_models": available,
          "embedding_models": embed_models,
          "matches": selected is not None,
      }
    except (
        httpx.HTTPError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
    ) as exc:
      return {
          "reachable": False,
          "served_model": None,
          "available_models": [],
          "embedding_models": [],
          "matches": False,
          "error": str(exc),
      }

  def verify(self) -> str:
    """Ensure endpoint is reachable and serves the expected model."""
    info = self.probe()
    if not info["reachable"]:
      detail = f": {info['error']}" if info.get("error") else ""
      raise EndpointUnreachable(f"cannot reach {self.base_url}{detail}")
    if not info["matches"]:
      avail = ", ".join(info["available_models"]) or "none"
      if self.model_id is None:
        raise ModelMismatch(
            f"no embedding model found at {self.base_url} "
            f"(available: {avail}); pass --model to specify one"
        )
      raise ModelMismatch(
          f"model {self.model_id!r} not found at {self.base_url} "
          f"(available: {avail})"
      )
    self.served_model = info["served_model"]
    self.model_id = canonical_model_id(self.served_model)
    self._verified = True
    return self.served_model

  def _build_image_payload(self, items: list[tuple[bytes, str]]) -> dict:
    content_items = []
    for raw, mime in items:
      b64 = base64.b64encode(raw).decode("ascii")
      url = f"data:{mime};base64,{b64}"
      content_items.append(
          {"content": [{"type": "image_url", "image_url": {"url": url}}]}
      )
    return {"model": self.served_model, "input": content_items}

  def embed_images(self, items: list[tuple[bytes, str]]) -> list[np.ndarray]:
    """Embed a batch of `(raw_bytes, mime_type)` image items."""
    if not items:
      return []
    if not self._verified:
      self.verify()
    payload = self._build_image_payload(items)
    try:
      vectors = self._post(payload, expected=len(items))
    except ItemRejected:
      if len(items) != 1:
        raise
      # On single-item fallback, re-encode via Pillow in case llama.cpp's
      # stb_image choked on an unusual valid image (e.g. 16-bit/CMYK/huge).
      reencoded = _transcode_to_png(items[0][0])
      if reencoded is None:
        raise
      retry_payload = self._build_image_payload([(reencoded, "image/png")])
      vectors = self._post(retry_payload, expected=1)
    return [normalize(v) for v in vectors]

  def embed_text(self, query: str) -> np.ndarray:
    """Embed a search query (applying EmbeddingGemma's prefix on Gemma)."""
    if not self._verified:
      self.verify()
    served_low = (self.served_model or "").lower()
    use_prefix = "embeddinggemma" in served_low
    prompt = (QUERY_PREFIX + query) if use_prefix else query
    vectors = self._post(
        {"model": self.served_model, "input": prompt}, expected=1
    )
    return normalize(vectors[0])

  def _post(self, payload: dict, expected: int) -> list:
    try:
      resp = self._client.post(self.base_url + EMBED_PATH, json=payload)
    except httpx.HTTPError as exc:
      raise EndpointUnreachable(str(exc)) from exc
    body_preview = resp.text[:200]
    low_body = body_preview.lower()
    if resp.status_code >= 500 and any(
        hint in low_body for hint in ITEM_REJECTED_HINTS
    ):
      raise ItemRejected(f"HTTP {resp.status_code}: {body_preview}")
    if resp.status_code >= 500 or resp.status_code in SYSTEMIC_HTTP_CODES:
      raise EndpointUnreachable(f"HTTP {resp.status_code}: {body_preview}")
    if resp.status_code >= 400:
      if "mmproj" in low_body or "image input is not supported" in low_body:
        raise EndpointUnreachable(
            f"server missing vision projector (HTTP {resp.status_code}): "
            f"{body_preview}"
        )
      raise ItemRejected(f"HTTP {resp.status_code}: {body_preview}")
    try:
      data = resp.json()["data"]
      if not isinstance(data, list):
        raise TypeError("data is not a list")
      if data and all(
          isinstance(it, dict) and isinstance(it.get("index"), int)
          for it in data
      ):
        data = sorted(data, key=lambda it: it["index"])
      vectors = [item["embedding"] for item in data]
    except (ValueError, KeyError, TypeError) as exc:
      raise EndpointUnreachable("malformed embedding response") from exc
    if len(vectors) != expected:
      raise EndpointUnreachable(
          f"expected {expected} vectors, got {len(vectors)}"
      )
    return vectors
