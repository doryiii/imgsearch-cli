"""Local Hugging Face / SentenceTransformers backend.

Weights are downloaded once into `./models/<model>` in cwd (or `--model-dir`)
and loaded directly from disk (`local_files_only=True`) with Selective Encoder
Loading (`vision_config=None, audio_config=None` for text queries;
`audio_config=None` for image indexing).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

from embed_base import (
    EmbedError,
    ItemRejected,
    QUERY_PREFIX,
    canonical_model_id,
    decode_rgb_image,
    is_embeddinggemma2,
    normalize,
    resolve_local_hf_model,
)

DEFAULT_MODELS_BASE = Path("models")
DEFAULT_MODEL_DIR = (
    Path(os.environ["EMBED_MODEL_DIR"])
    if os.environ.get("EMBED_MODEL_DIR")
    else None
)
DEFAULT_DEVICE = os.environ.get("EMBED_DEVICE") or None
INSTALL_HINT = (
    "local mode requires `sentence-transformers>=6.1.0`, "
    "`transformers>=5.18.0`, `safetensors`, `huggingface-hub`, and `torch` "
    "(install with `uv pip install -e '.[local]'`)"
)


def _ensure_venv_path() -> None:
  """Add sibling `.venv` site-packages to `sys.path` if needed."""
  venv_root = Path(__file__).resolve().parent / ".venv" / "lib"
  py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
  candidate = venv_root / py_ver / "site-packages"
  if candidate.is_dir() and str(candidate) not in sys.path:
    sys.path.insert(0, str(candidate))


def _import_local_ml():
  """Import `torch` and `SentenceTransformer`, checking `.venv` if needed."""
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
      raise EmbedError(INSTALL_HINT) from exc


def _select_device_and_dtype(torch_mod, requested_device: str | None):
  """Pick device and safe dtype (`bfloat16` on bf16 CUDA, `float32` on CPU)."""
  try:
    cuda_ok = bool(torch_mod.cuda.is_available())
    if requested_device:
      device = requested_device
      if device.startswith("cuda") and not cuda_ok:
        raise EmbedError(
            f"requested device {device!r}, but CUDA is not available"
        )
    else:
      device = "cuda" if cuda_ok else "cpu"

    if (
        device.startswith("cuda")
        and cuda_ok
        and torch_mod.cuda.is_bf16_supported()
    ):
      dtype = torch_mod.bfloat16
    else:
      dtype = torch_mod.float32
    return device, dtype
  except EmbedError:
    raise
  except Exception as exc:
    raise EmbedError(
        f"invalid torch device {requested_device!r}: {exc}"
    ) from exc


def has_local_weights(weights_dir: Path) -> bool:
  """Return True when `weights_dir` holds complete config and weight files."""
  if not (weights_dir / "config.json").is_file():
    return False
  index_file = weights_dir / "model.safetensors.index.json"
  if index_file.is_file():
    try:
      idx_data = json.loads(index_file.read_text(encoding="utf-8"))
      weight_map = idx_data.get("weight_map")
      if not isinstance(weight_map, dict) or not weight_map:
        return False
      shards = {str(v) for v in weight_map.values() if v}
      return bool(shards) and all(
          (weights_dir / s).is_file() and (weights_dir / s).stat().st_size > 0
          for s in shards
      )
    except (OSError, ValueError):
      return False

  for pattern in ("*.safetensors", "*.bin"):
    for f in weights_dir.glob(pattern):
      try:
        if f.is_file() and f.stat().st_size > 0:
          return True
      except OSError:
        continue
  return False


def resolve_weights_dir(
    hf_model_id: str, model_dir: str | Path | None = None
) -> Path:
  """Resolve local directory where model weights are stored and loaded from."""
  candidate_path = Path(hf_model_id)
  if candidate_path.is_dir() and has_local_weights(candidate_path):
    return candidate_path

  effective_dir = model_dir if model_dir is not None else DEFAULT_MODEL_DIR
  if effective_dir is not None:
    return Path(effective_dir)

  slug = hf_model_id.rstrip("/").split("/")[-1]
  return DEFAULT_MODELS_BASE / slug


def _is_gemma2_weights(
    hf_model_id: str, weights_dir: Path | None = None
) -> bool:
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
  """Build `config_kwargs` for EmbeddingGemma 2 Selective Encoder Loading."""
  if not _is_gemma2_weights(hf_model_id, weights_dir):
    return {}
  if need_vision:
    return {"audio_config": None}
  return {"vision_config": None, "audio_config": None}


def ensure_weights_downloaded(hf_model_id: str, weights_dir: Path) -> Path:
  """Download `hf_model_id` into `weights_dir` if not already present."""
  if has_local_weights(weights_dir):
    return weights_dir

  if (
      hf_model_id.startswith((".", "/", "~"))
      and not Path(hf_model_id).expanduser().is_dir()
  ):
    raise EmbedError(f"local model directory not found: {hf_model_id}")

  _ensure_venv_path()
  try:
    from huggingface_hub import snapshot_download
  except ImportError as exc:
    raise EmbedError(
        "model download requires `huggingface-hub` "
        "(`uv pip install huggingface-hub`)"
    ) from exc

  print(
      f"Downloading {hf_model_id} weights to {weights_dir} ...", file=sys.stderr
  )
  weights_dir.mkdir(parents=True, exist_ok=True)
  try:
    snapshot_download(
        repo_id=hf_model_id,
        local_dir=str(weights_dir),
        cache_dir=str(weights_dir / ".cache"),
    )
  except Exception as exc:
    raise EmbedError(
        f"failed to download {hf_model_id!r} to {weights_dir}: {exc}"
    ) from exc
  return weights_dir


class LocalEmbeddingClient:
  """In-process backend loading weights from `./models/<model>` on disk."""

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
    self.model_id: str | None = canonical_model_id(
        model_id if model_id is not None else self.hf_model_id
    )
    self.served_model: str | None = self.hf_model_id
    self.mode = mode
    self.device = device
    self._loader = loader
    self._model = None
    self._loaded_vision = False
    self._loaded_hf_id: str | None = None

  @property
  def weights_dir(self) -> Path:
    return resolve_weights_dir(self.hf_model_id, self.model_dir)

  def close(self) -> None:
    self._model = None
    self._loaded_vision = False
    self._loaded_hf_id = None

  def __enter__(self) -> LocalEmbeddingClient:
    return self

  def __exit__(self, *exc) -> None:
    self.close()

  def probe(self) -> dict:
    """Report local backend status without eagerly loading weights into RAM."""
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
    """Verify local ML packages are importable (weights load lazily)."""
    if self._loader is None:
      _import_local_ml()
    if self.model_id is None:
      self.model_id = canonical_model_id(self.hf_model_id)
    self.served_model = self.hf_model_id
    return self.served_model

  def _ensure_loaded(self, need_vision: bool):
    if (
        self._model is not None
        and self._loaded_hf_id == self.hf_model_id
        and (self._loaded_vision or not need_vision)
    ):
      return self._model

    # Drop any prior text-only model before allocating a larger vision model.
    self._model = None
    target_vision = need_vision or (self.mode == "vision")
    wdir = self.weights_dir
    if self._loader is not None:
      self._model = self._loader(
          self.hf_model_id, wdir, target_vision, self.device
      )
      self._loaded_vision = target_vision
      self._loaded_hf_id = self.hf_model_id
      return self._model

    torch_mod, st_cls = _import_local_ml()
    try:
      import transformers

      transformers.utils.logging.disable_progress_bar()
    except Exception:
      pass

    resolved_device, dtype = _select_device_and_dtype(torch_mod, self.device)
    ensure_weights_downloaded(self.hf_model_id, wdir)
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
    self._loaded_hf_id = self.hf_model_id
    return self._model

  def embed_text(self, query: str) -> np.ndarray:
    """Encode a search query using only the text encoder when `mode='text'`."""
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
    elif "query" in prompts:
      vec = model.encode(
          query,
          prompt_name="query",
          normalize_embeddings=True,
          convert_to_numpy=True,
          show_progress_bar=False,
      )
    else:
      use_prefix = (
          _is_gemma2_weights(self.hf_model_id, self.weights_dir)
          or "embeddinggemma" in self.hf_model_id.lower()
      )
      text = (QUERY_PREFIX + query) if use_prefix else query
      vec = model.encode(
          text,
          normalize_embeddings=True,
          convert_to_numpy=True,
          show_progress_bar=False,
      )
    return normalize(vec)

  def embed_images(self, items: list[tuple[bytes, str]]) -> list[np.ndarray]:
    """Encode `(raw_bytes, mime)` images using text + vision encoders."""
    if not items:
      return []
    pil_inputs: list[dict] = []
    for raw, _mime in items:
      try:
        rgb = decode_rgb_image(raw)
      except EmbedError:
        raise
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
      if len(pil_inputs) > 1:
        raise ItemRejected(f"batch image encoding failed: {exc}") from exc
      raise EmbedError(f"local image encoding failed: {exc}") from exc

    return [normalize(v) for v in vecs]
