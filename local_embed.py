"""Local Hugging Face / SentenceTransformers backend with Selective Encoder Loading.

Weights are downloaded once into a local directory (`./models/<model>` in cwd by default,
or `--model-dir`) via `huggingface_hub.snapshot_download(local_dir=...)` and loaded
directly from that directory (`local_files_only=True`) rather than `~/.cache/huggingface`.

For `google/embeddinggemma-2`:
- Text-only queries (`mode="text"`) pass `config_kwargs={"vision_config": None, "audio_config": None}`
  to load only the ~270M text backbone + embedder on CPU/GPU.
- Image indexing / visual similarity (`mode="vision"`) passes `config_kwargs={"audio_config": None}`
  to load only text + vision (~440M), skipping the ~300M audio tower.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys

import numpy as np

from embed_base import (
    EmbedError,
    ItemRejected,
    QUERY_PREFIX,
    is_embeddinggemma2,
    normalize,
    resolve_local_hf_model,
)

DEFAULT_MODELS_BASE = Path("models")
DEFAULT_MODEL_DIR = (
    Path(os.environ["EMBED_MODEL_DIR"]) if os.environ.get("EMBED_MODEL_DIR") else None
)
DEFAULT_DEVICE = os.environ.get("EMBED_DEVICE") or None


def _ensure_venv_path() -> None:
  """Add sibling `.venv` site-packages to `sys.path` if running under bare system Python."""
  venv_root = Path(__file__).resolve().parent / ".venv" / "lib"
  py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
  candidate = venv_root / py_ver / "site-packages"
  if candidate.is_dir() and str(candidate) not in sys.path:
    sys.path.insert(0, str(candidate))


def _import_local_ml():
  """Import `torch` and `SentenceTransformer`, checking sibling `.venv` if needed."""
  try:
    import torch
    from sentence_transformers import SentenceTransformer

    return torch, SentenceTransformer
  except ImportError:
    _ensure_venv_path()
    try:
      import torch
      from sentence_transformers import SentenceTransformer

      return torch, SentenceTransformer
    except ImportError as exc:
      raise EmbedError(
          "local mode requires `sentence-transformers>=6.1.0` and `transformers` "
          "(install with `uv pip install sentence-transformers transformers safetensors`)"
      ) from exc


def _select_device_and_dtype(torch_mod, requested_device: str | None):
  """Pick execution device and safe precision (`bfloat16` on bf16 CUDA, `float32` on CPU)."""
  if requested_device:
    device = requested_device
  else:
    device = "cuda" if torch_mod.cuda.is_available() else "cpu"

  # EmbeddingGemma 2 overflows float16; use bfloat16 on supported CUDA GPUs and float32 on CPU.
  if device.startswith("cuda") and torch_mod.cuda.is_bf16_supported():
    dtype = torch_mod.bfloat16
  else:
    dtype = torch_mod.float32
  return device, dtype


def has_local_weights(weights_dir: Path) -> bool:
  """Return True when `weights_dir` already contains downloaded model config and weights."""
  if not (weights_dir / "config.json").is_file():
    return False
  return (
      any(weights_dir.glob("*.safetensors"))
      or (weights_dir / "model.safetensors.index.json").is_file()
      or any(weights_dir.glob("*.bin"))
  )


def resolve_weights_dir(hf_model_id: str, model_dir: str | Path | None = None) -> Path:
  """Resolve on-disk directory where model weights should be stored and loaded from."""
  candidate_path = Path(hf_model_id)
  if candidate_path.is_dir() and has_local_weights(candidate_path):
    return candidate_path

  effective_dir = model_dir if model_dir is not None else DEFAULT_MODEL_DIR
  if effective_dir is not None:
    return Path(effective_dir)

  slug = hf_model_id.rstrip("/").split("/")[-1]
  return DEFAULT_MODELS_BASE / slug


def _is_gemma2_weights(hf_model_id: str, weights_dir: Path | None = None) -> bool:
  if is_embeddinggemma2(hf_model_id):
    return True
  if weights_dir is not None:
    cfg_file = weights_dir / "config.json"
    if cfg_file.is_file():
      try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
      except (OSError, ValueError):
        return False
      if cfg.get("model_type") == "embedding_gemma2":
        return True
      if "EmbeddingGemma2Model" in (cfg.get("architectures") or []):
        return True
  return False


def build_selective_config_kwargs(
    hf_model_id: str, need_vision: bool, weights_dir: Path | None = None
) -> dict:
  """Build `SentenceTransformer(..., config_kwargs=...)` for Selective Encoder Loading."""
  if not _is_gemma2_weights(hf_model_id, weights_dir):
    return {}
  if need_vision:
    return {"audio_config": None}
  return {"vision_config": None, "audio_config": None}


def ensure_weights_downloaded(hf_model_id: str, weights_dir: Path) -> Path:
  """Download `hf_model_id` directly into `weights_dir` if not already present."""
  if has_local_weights(weights_dir):
    return weights_dir

  _ensure_venv_path()
  try:
    from huggingface_hub import snapshot_download
  except ImportError as exc:
    raise EmbedError(
        "local model download requires `huggingface_hub` (`uv pip install huggingface-hub`)"
    ) from exc

  print(f"Downloading {hf_model_id} weights to {weights_dir} ...", file=sys.stderr)
  weights_dir.mkdir(parents=True, exist_ok=True)
  try:
    snapshot_download(
        repo_id=hf_model_id,
        local_dir=str(weights_dir),
        cache_dir=str(weights_dir / ".cache"),
    )
  except Exception as exc:
    raise EmbedError(f"failed to download {hf_model_id!r} to {weights_dir}: {exc}") from exc
  return weights_dir


class LocalEmbeddingClient:
  """In-process embedding backend loading weights from `./models/<model>` (or `--model-dir`)."""

  is_local = True

  def __init__(
      self,
      model_id: str | None = None,
      model_dir: str | Path | None = DEFAULT_MODEL_DIR,
      mode: str = "text",
      device: str | None = DEFAULT_DEVICE,
      loader=None,
  ):
    if mode not in ("text", "vision"):
      raise ValueError(f"mode must be 'text' or 'vision', got {mode!r}")
    self._explicit_model = model_id is not None
    self.hf_model_id = resolve_local_hf_model(model_id)
    self.model_dir = Path(model_dir) if model_dir is not None else None
    # Canonical model_id defaults to 'embeddinggemma-2' for EmbeddingGemma 2 so indexes are
    # seamlessly shared between a remote llama.cpp server and local runs.
    if model_id is not None:
      self.model_id: str | None = model_id
    elif is_embeddinggemma2(self.hf_model_id):
      self.model_id = "embeddinggemma-2"
    else:
      self.model_id = self.hf_model_id

    self.served_model: str | None = self.hf_model_id
    self.mode = mode
    self.device = device
    self._loader = (
        loader  # optional test hook: (hf_model_id, weights_dir, need_vision, device) -> model
    )
    self._model = None
    self._loaded_vision = False

  @property
  def weights_dir(self) -> Path:
    return resolve_weights_dir(self.hf_model_id, self.model_dir)

  def close(self) -> None:
    self._model = None
    self._loaded_vision = False

  def __enter__(self) -> LocalEmbeddingClient:
    return self

  def __exit__(self, *exc) -> None:
    self.close()

  def probe(self) -> dict:
    """Report local backend status and weights directory without loading weights into RAM."""
    if self._loader is not None:
      reachable = True
      err = None
    else:
      try:
        _import_local_ml()
        reachable = True
        err = None
      except EmbedError as exc:
        reachable = False
        err = str(exc)
    wdir = self.weights_dir
    result = {
        "reachable": reachable,
        "served_model": self.hf_model_id if reachable else None,
        "available_models": [self.hf_model_id] if reachable else [],
        "embedding_models": [self.hf_model_id] if reachable else [],
        "matches": reachable,
        "local": True,
        "selective_mode": self.mode,
        "weights_dir": str(wdir),
        "weights_downloaded": has_local_weights(wdir),
    }
    if err:
      result["error"] = err
    return result

  def verify(self) -> str:
    """Verify local ML dependencies are importable (weights remain lazily downloaded/loaded)."""
    if self._loader is None:
      _import_local_ml()
    if self.model_id is None:
      self.model_id = (
          "embeddinggemma-2" if is_embeddinggemma2(self.hf_model_id) else self.hf_model_id
      )
    self.served_model = self.hf_model_id
    return self.served_model

  def _ensure_loaded(self, need_vision: bool):
    if self._model is not None and (self._loaded_vision or not need_vision):
      return self._model

    target_vision = need_vision or (self.mode == "vision")
    wdir = self.weights_dir
    if self._loader is not None:
      self._model = self._loader(self.hf_model_id, wdir, target_vision, self.device)
      self._loaded_vision = target_vision
      return self._model

    torch_mod, st_cls = _import_local_ml()
    try:
      import transformers

      transformers.utils.logging.disable_progress_bar()
    except Exception:
      pass

    ensure_weights_downloaded(self.hf_model_id, wdir)
    resolved_device, dtype = _select_device_and_dtype(torch_mod, self.device)
    config_kwargs = build_selective_config_kwargs(
        self.hf_model_id, need_vision=target_vision, weights_dir=wdir
    )
    try:
      self._model = st_cls(
          str(wdir),
          device=resolved_device,
          local_files_only=True,
          model_kwargs={"dtype": dtype},
          config_kwargs=config_kwargs,
      )
    except Exception as exc:
      raise EmbedError(
          f"failed to load local model from {wdir} ({self.hf_model_id}): {exc}"
      ) from exc

    self._loaded_vision = target_vision
    return self._model

  def embed_text(self, query: str) -> np.ndarray:
    """Encode a text search query using only the text encoder when `mode='text'`."""
    model = self._ensure_loaded(need_vision=False)
    prompts = getattr(model, "prompts", None) or {}
    if "SearchQuery" in prompts:
      vec = model.encode(
          query,
          prompt_name="SearchQuery",
          normalize_embeddings=True,
          convert_to_numpy=True,
          show_progress_bar=False,
      )
    else:
      vec = model.encode(
          QUERY_PREFIX + query,
          normalize_embeddings=True,
          convert_to_numpy=True,
          show_progress_bar=False,
      )
    return normalize(vec)

  def embed_images(self, items: list[tuple[bytes, str]]) -> list[np.ndarray]:
    """Encode a batch of `(raw_bytes, mime)` images using text + vision encoders."""
    if not items:
      return []
    try:
      from PIL import Image, ImageOps
    except ImportError as exc:
      raise EmbedError("Pillow is required to decode images for local embedding") from exc

    pil_inputs: list[dict] = []
    for raw, _mime in items:
      try:
        with Image.open(io.BytesIO(raw)) as im:
          rgb = ImageOps.exif_transpose(im).convert("RGB")
      except Exception as exc:
        raise ItemRejected(f"invalid image payload: {exc}") from exc
      pil_inputs.append({"image": rgb})

    model = self._ensure_loaded(need_vision=True)
    try:
      vecs = model.encode(
          pil_inputs,
          batch_size=len(pil_inputs),
          normalize_embeddings=True,
          convert_to_numpy=True,
          show_progress_bar=False,
      )
    except Exception as exc:
      raise ItemRejected(f"local image encoding failed: {exc}") from exc

    return [normalize(v) for v in vecs]
