"""Remote HTTP embedding backend for llama.cpp / OpenAI-compatible `/v1/embeddings` servers."""

from __future__ import annotations

import base64
import os

import httpx
import numpy as np

from embed_base import (
    EndpointUnreachable,
    ItemRejected,
    ModelMismatch,
    QUERY_PREFIX,
    normalize,
)

DEFAULT_ENDPOINT = os.environ.get("EMBED_ENDPOINT") or None
DEFAULT_MODEL = os.environ.get("EMBED_MODEL") or None

EMBED_PATH = "/v1/embeddings"
MODELS_PATH = "/v1/models"
PROBE_TIMEOUT = 5.0
REQUEST_TIMEOUT = 300.0

EMBED_NAME_TOKENS = ("embed", "bge-", "bge_", "e5-", "gte-", "siglip", "jina-clip")


def _parse_models(payload: dict) -> list[dict]:
  """Normalize OpenAI `data[]` and llama.cpp/Ollama `models[]` entries in list order."""
  raw_data = payload.get("data") if isinstance(payload.get("data"), list) else []
  raw_models = payload.get("models") if isinstance(payload.get("models"), list) else []

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
        if raw_data and idx < len(raw_models) and isinstance(raw_models[idx], dict)
        else {}
    )
    arch = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    details = companion.get("details") if isinstance(companion.get("details"), dict) else {}
    raw_caps = companion.get("capabilities") or item.get("capabilities") or []
    raw_tags = (item.get("tags") or []) + (companion.get("tags") or [])
    raw_aliases = item.get("aliases") or companion.get("aliases") or []
    raw_args = status.get("args") or []

    parsed.append(
        {
            "id": model_id,
            "aliases": [str(a) for a in raw_aliases if a],
            "capabilities": [str(c).lower() for c in raw_caps if c],
            "type": str(companion.get("type") or item.get("type") or "").lower(),
            "tags": [str(t).lower() for t in raw_tags if t],
            "output_modalities": [
                str(x).lower() for x in (arch.get("output_modalities") or []) if x
            ],
            "status_args": [str(a).lower() for a in raw_args if a],
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
    """Query GET /v1/models for reachability and resolve target or first embedding model."""
    try:
      resp = self._client.get(self.base_url + MODELS_PATH, timeout=PROBE_TIMEOUT)
      resp.raise_for_status()
      entries = _parse_models(resp.json())
      available = [e["id"] for e in entries]
      embed_models = [e["id"] for e in entries if _is_embedding_model(e)]

      if self.model_id:
        needle = self.model_id.lower()
        matched = next(
            (
                e["id"]
                for e in entries
                if needle in e["id"].lower() or any(needle in a.lower() for a in e["aliases"])
            ),
            None,
        )
        return {
            "reachable": True,
            "served_model": matched or (available[0] if available else None),
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
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
      return {
          "reachable": False,
          "served_model": None,
          "available_models": [],
          "embedding_models": [],
          "matches": False,
          "error": str(exc),
      }

  def verify(self) -> str:
    """Ensure endpoint is reachable and serves the expected (or an embedding) model."""
    info = self.probe()
    if not info["reachable"]:
      detail = f": {info['error']}" if info.get("error") else ""
      raise EndpointUnreachable(f"cannot reach {self.base_url}{detail}")
    if not info["matches"]:
      avail = ", ".join(info["available_models"]) or "none"
      if self.model_id is None:
        raise ModelMismatch(
            f"no embedding model found at {self.base_url} (available: {avail}); "
            "pass --model to specify one"
        )
      raise ModelMismatch(
          f"model {self.model_id!r} not found at {self.base_url} (available: {avail})"
      )
    self.served_model = info["served_model"]
    if self.model_id is None:
      self.model_id = self.served_model
    self._verified = True
    return self.served_model

  def embed_images(self, items: list[tuple[bytes, str]]) -> list[np.ndarray]:
    """Embed a batch of `(raw_bytes, mime_type)` image items."""
    if not items:
      return []
    if not self._verified:
      self.verify()
    payload = {
        "model": self.served_model,
        "input": [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
                        },
                    }
                ]
            }
            for raw, mime in items
        ],
    }
    vectors = self._post(payload, expected=len(items))
    return [normalize(v) for v in vectors]

  def embed_text(self, query: str) -> np.ndarray:
    """Embed a search query using EmbeddingGemma's query prefix."""
    if not self._verified:
      self.verify()
    vectors = self._post({"model": self.served_model, "input": QUERY_PREFIX + query}, expected=1)
    return normalize(vectors[0])

  def _post(self, payload: dict, expected: int) -> list:
    try:
      resp = self._client.post(self.base_url + EMBED_PATH, json=payload)
    except httpx.HTTPError as exc:
      raise EndpointUnreachable(str(exc)) from exc
    if resp.status_code >= 500 or resp.status_code in (408, 429):
      raise EndpointUnreachable(f"HTTP {resp.status_code}: {resp.text[:200]}")
    if resp.status_code >= 400:
      raise ItemRejected(f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
      data = resp.json()["data"]
      vectors = [item["embedding"] for item in data]
    except (ValueError, KeyError, TypeError) as exc:
      raise EndpointUnreachable("malformed embedding response") from exc
    if len(vectors) != expected:
      raise EndpointUnreachable(f"expected {expected} vectors, got {len(vectors)}")
    return vectors
