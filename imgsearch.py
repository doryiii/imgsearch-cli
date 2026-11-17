#!/usr/bin/env python3
"""Vector indexer and search CLI for images using EmbeddingGemma 2 (llama.cpp).

- Image documents are sent nested inside `input[].content[].image_url`.
- Text search queries use EmbeddingGemma's asymmetric retrieval prefix:
  `task: search result | query: <query>`
- Returned vectors are L2-normalized float32 and stored in SQLite (`index.db`).
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from urllib.request import pathname2url

import httpx
import numpy as np

DEFAULT_ENDPOINT = os.environ.get("EMBED_ENDPOINT", "http://localhost:8080")
DEFAULT_MODEL = os.environ.get("EMBED_MODEL") or None
DEFAULT_ROOT = Path(".")
DEFAULT_DB = Path("index.db")

EMBED_PATH = "/v1/embeddings"
MODELS_PATH = "/v1/models"
QUERY_PREFIX = "task: search result | query: "
PROBE_TIMEOUT = 5.0
REQUEST_TIMEOUT = 300.0
DEFAULT_BATCH_SIZE = 8

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
EXT_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
SUBCOMMANDS = {"index", "search", "similar", "status"}
THUMB_BACKENDS = ("auto", "kitty", "sixel", "ansi")
DEFAULT_THUMB_ROWS = 10
THUMB_INDENT = "    "


class EmbedError(Exception):
  """Base error for embedding endpoint failures."""


class EndpointUnreachable(EmbedError):
  """Connection error, timeout, 5xx, or malformed response."""


class ModelMismatch(EmbedError):
  """Endpoint is serving a model that does not match configured model_id."""


class ItemRejected(EmbedError):
  """Endpoint rejected the payload with a 4xx status."""


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
      base_url: str = DEFAULT_ENDPOINT,
      model_id: str | None = DEFAULT_MODEL,
      transport=None,
  ):
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


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    image_id    INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    sha256      TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    model_id    TEXT,
    dim         INTEGER,
    vector      BLOB,
    embedded_at REAL,
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);
"""


def open_db(db_path: Path, readonly: bool = False) -> sqlite3.Connection:
  db_path = Path(db_path)
  if readonly:
    url_path = pathname2url(str(db_path.resolve()))
    conn: sqlite3.Connection | None = None
    try:
      conn = sqlite3.connect(f"file:{url_path}?mode=ro", uri=True, timeout=30)
      conn.row_factory = sqlite3.Row
      conn.execute("PRAGMA busy_timeout=5000")
      conn.execute("PRAGMA schema_version").fetchone()
      return conn
    except sqlite3.OperationalError:
      if conn is not None:
        conn.close()
      # Fallback for WAL-flagged DBs in directories where the caller cannot create -shm/-wal files
      conn = sqlite3.connect(f"file:{url_path}?mode=ro&immutable=1", uri=True, timeout=30)
      conn.row_factory = sqlite3.Row
      return conn

  db_path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(db_path, timeout=30)
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA journal_mode=DELETE")
  conn.execute("PRAGMA synchronous=NORMAL")
  conn.execute("PRAGMA busy_timeout=5000")
  conn.executescript(SCHEMA_SQL)
  return conn


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
  row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
  return row["value"] if row is not None else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
  conn.execute(
      "INSERT INTO meta(key, value) VALUES(?, ?) "
      "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      (key, value),
  )


def discover_images(root: Path, db_path: Path) -> list[Path]:
  """Recursively find image files under `root`, skipping hidden files/dirs."""
  db_resolved = db_path.resolve()
  found: list[Path] = []
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
    for fname in sorted(filenames):
      if fname.startswith("."):
        continue
      p = Path(dirpath) / fname
      if p.suffix.lower() not in IMAGE_EXTENSIONS:
        continue
      if p.resolve() == db_resolved:
        continue
      found.append(p)
  return found


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def format_duration(seconds: float) -> str:
  if seconds < 60:
    return f"{seconds:.1f}s"
  mins, secs = divmod(int(round(seconds)), 60)
  if mins < 60:
    return f"{mins}m{secs:02d}s"
  hrs, mins = divmod(mins, 60)
  return f"{hrs}h{mins:02d}m"


def build_index(
    root: Path,
    db_path: Path,
    client: EmbeddingClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    retry_errors: bool = False,
    keep_missing: bool = False,
    quiet: bool = False,
) -> dict:
  """Scan `root`, reuse unchanged/duplicate embeddings, and embed new/updated images."""
  root = root.resolve()
  if not root.is_dir():
    raise FileNotFoundError(f"root directory does not exist: {root}")

  served_model = client.verify()
  conn = open_db(db_path)
  try:
    with conn:
      set_meta(conn, "root_dir", str(root))
      set_meta(conn, "model_id", client.model_id)
      set_meta(conn, "served_model", served_model)
      set_meta(conn, "endpoint", client.base_url)

    files = discover_images(root, db_path)
    existing_rows = {row["path"]: row for row in conn.execute("SELECT * FROM images").fetchall()}
    # Cache known vectors by sha256 for the active model so duplicates/moves don't re-embed
    hash_vectors: dict[str, tuple[int, bytes]] = {}
    if not force:
      for row in existing_rows.values():
        if row["vector"] is not None and row["model_id"] == client.model_id:
          hash_vectors[row["sha256"]] = (int(row["dim"]), bytes(row["vector"]))

    seen_rel_paths: set[str] = set()
    up_to_date = 0
    reused_hash = 0
    skipped_errors = 0
    failed = 0
    to_embed: list[tuple[str, Path, int, int, str]] = (
        []
    )  # (rel_path, full_path, size, mtime_ns, sha256)

    for full_path in files:
      rel_path = full_path.relative_to(root).as_posix()
      seen_rel_paths.add(rel_path)
      try:
        st = full_path.stat()
      except OSError as exc:
        _record_error(conn, rel_path, "", 0, 0, client.model_id, str(exc))
        failed += 1
        continue

      prev = existing_rows.get(rel_path)

      if (
          not force
          and prev is not None
          and prev["size"] == st.st_size
          and prev["mtime_ns"] == st.st_mtime_ns
          and prev["model_id"] == client.model_id
      ):
        if prev["vector"] is not None:
          up_to_date += 1
          continue
        if prev["error"] is not None and not retry_errors:
          # Automatically retry if a previously unreadable file became readable
          if "Permission denied" in prev["error"] and os.access(full_path, os.R_OK):
            pass
          else:
            skipped_errors += 1
            continue

      try:
        raw = full_path.read_bytes()
      except OSError as exc:
        _record_error(conn, rel_path, "", st.st_size, st.st_mtime_ns, client.model_id, str(exc))
        failed += 1
        continue

      digest = sha256_bytes(raw)
      if not force and digest in hash_vectors:
        dim, vec_bytes = hash_vectors[digest]
        with conn:
          conn.execute(
              """
              INSERT INTO images(
                  path, sha256, size, mtime_ns, model_id, dim,
                  vector, embedded_at, error
              )
              VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL)
              ON CONFLICT(path) DO UPDATE SET
                  sha256=excluded.sha256,
                  size=excluded.size,
                  mtime_ns=excluded.mtime_ns,
                  model_id=excluded.model_id,
                  dim=excluded.dim,
                  vector=excluded.vector,
                  embedded_at=excluded.embedded_at,
                  error=NULL
              """,
              (
                  rel_path,
                  digest,
                  st.st_size,
                  st.st_mtime_ns,
                  client.model_id,
                  dim,
                  vec_bytes,
                  time.time(),
              ),
          )
        reused_hash += 1
        continue

      to_embed.append((rel_path, full_path, st.st_size, st.st_mtime_ns, digest))

    pruned = 0
    if not keep_missing:
      missing = set(existing_rows.keys()) - seen_rel_paths
      if missing:
        with conn:
          conn.executemany("DELETE FROM images WHERE path = ?", [(p,) for p in missing])
        pruned = len(missing)

    total_todo = len(to_embed)
    if not quiet:
      extra = []
      if failed:
        extra.append(f"unreadable: {failed}")
      if skipped_errors:
        extra.append(f"prev-errors: {skipped_errors}")
      if pruned:
        extra.append(f"pruned: {pruned}")
      extra_str = (", " + ", ".join(extra)) if extra else ""
      print(
          f"Found {len(files)} images in {root} "
          f"(up-to-date: {up_to_date}, hash-reused: {reused_hash}, to-embed: {total_todo}"
          f"{extra_str})",
          file=sys.stderr,
      )

    embedded = 0
    t_start = time.perf_counter()

    for start in range(0, total_todo, batch_size):
      chunk = to_embed[start : start + batch_size]
      loaded: list[tuple[tuple[str, Path, int, int, str], bytes, str]] = []
      for entry in chunk:
        rel_path, full_path, size, mtime_ns, digest = entry
        try:
          raw = full_path.read_bytes()
        except OSError as exc:
          _record_error(conn, rel_path, digest, size, mtime_ns, client.model_id, str(exc))
          failed += 1
          continue
        # Check if an earlier item in this run already embedded the exact same sha256
        if not force and digest in hash_vectors:
          dim, vec_bytes = hash_vectors[digest]
          _record_vector(conn, rel_path, digest, size, mtime_ns, client.model_id, dim, vec_bytes)
          reused_hash += 1
          continue
        loaded.append((entry, raw, detect_mime(raw, full_path)))

      if loaded:
        try:
          vectors = client.embed_images([(raw, mime) for _, raw, mime in loaded])
          now_ts = time.time()
          with conn:
            for (rel_path, _, size, mtime_ns, digest), vec in zip(
                (item[0] for item in loaded), vectors, strict=True
            ):
              vec_f32 = np.asarray(vec, dtype="<f4")
              vec_bytes = vec_f32.tobytes()
              dim = int(vec_f32.shape[0])
              hash_vectors[digest] = (dim, vec_bytes)
              conn.execute(
                  """
                  INSERT INTO images(
                      path, sha256, size, mtime_ns, model_id, dim,
                      vector, embedded_at, error
                  )
                  VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL)
                  ON CONFLICT(path) DO UPDATE SET
                      sha256=excluded.sha256,
                      size=excluded.size,
                      mtime_ns=excluded.mtime_ns,
                      model_id=excluded.model_id,
                      dim=excluded.dim,
                      vector=excluded.vector,
                      embedded_at=excluded.embedded_at,
                      error=NULL
                  """,
                  (rel_path, digest, size, mtime_ns, client.model_id, dim, vec_bytes, now_ts),
              )
              embedded += 1
        except ItemRejected:
          # Fallback to per-item requests so one bad image doesn't reject the whole batch
          for (rel_path, _, size, mtime_ns, digest), raw, mime in loaded:
            try:
              vec = client.embed_images([(raw, mime)])[0]
              vec_f32 = np.asarray(vec, dtype="<f4")
              vec_bytes = vec_f32.tobytes()
              dim = int(vec_f32.shape[0])
              hash_vectors[digest] = (dim, vec_bytes)
              _record_vector(
                  conn, rel_path, digest, size, mtime_ns, client.model_id, dim, vec_bytes
              )
              embedded += 1
            except ItemRejected as exc:
              _record_error(conn, rel_path, digest, size, mtime_ns, client.model_id, str(exc))
              failed += 1

      if not quiet and total_todo > 0:
        done = min(start + len(chunk), total_todo)
        elapsed = max(time.perf_counter() - t_start, 1e-6)
        rate = done / elapsed
        eta = (total_todo - done) / rate if rate > 0 else 0.0
        pct = 100.0 * done / total_todo
        msg = (
            f"\r[{done}/{total_todo}] {pct:5.1f}% | "
            f"{rate:5.1f} img/s | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}   "
        )
        sys.stderr.write(msg)
        sys.stderr.flush()

    if not quiet and total_todo > 0:
      sys.stderr.write("\n")
      sys.stderr.flush()

    elapsed_total = time.perf_counter() - t_start
    with conn:
      set_meta(conn, "last_indexed_at", str(time.time()))

    return {
        "total_files": len(files),
        "up_to_date": up_to_date,
        "reused_hash": reused_hash,
        "embedded": embedded,
        "failed": failed,
        "skipped_errors": skipped_errors,
        "pruned": pruned,
        "elapsed_seconds": round(elapsed_total, 2),
    }
  finally:
    conn.close()


def _record_vector(
    conn: sqlite3.Connection,
    rel_path: str,
    digest: str,
    size: int,
    mtime_ns: int,
    model_id: str,
    dim: int,
    vec_bytes: bytes,
) -> None:
  with conn:
    conn.execute(
        """
        INSERT INTO images(path, sha256, size, mtime_ns, model_id, dim, vector, embedded_at, error)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(path) DO UPDATE SET
            sha256=excluded.sha256,
            size=excluded.size,
            mtime_ns=excluded.mtime_ns,
            model_id=excluded.model_id,
            dim=excluded.dim,
            vector=excluded.vector,
            embedded_at=excluded.embedded_at,
            error=NULL
        """,
        (rel_path, digest, size, mtime_ns, model_id, dim, vec_bytes, time.time()),
    )


def _record_error(
    conn: sqlite3.Connection,
    rel_path: str,
    digest: str,
    size: int,
    mtime_ns: int,
    model_id: str,
    error: str,
) -> None:
  with conn:
    conn.execute(
        """
        INSERT INTO images(path, sha256, size, mtime_ns, model_id, dim, vector, embedded_at, error)
        VALUES(?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
        ON CONFLICT(path) DO UPDATE SET
            sha256=excluded.sha256,
            size=excluded.size,
            mtime_ns=excluded.mtime_ns,
            model_id=excluded.model_id,
            vector=NULL,
            error=excluded.error
        """,
        (rel_path, digest, size, mtime_ns, model_id, error),
    )


def load_resident_index(
    conn: sqlite3.Connection,
    model_id: str,
    folder: str | None = None,
) -> tuple[Path, list[dict], np.ndarray | None]:
  """Load stored float32 vectors from SQLite into a single contiguous `(N, dim)` matrix."""
  root_str = get_meta(conn, "root_dir")
  root_dir = Path(root_str) if root_str else DEFAULT_ROOT

  sql = (
      "SELECT image_id, path, sha256, size, mtime_ns, dim, vector "
      "FROM images WHERE vector IS NOT NULL AND model_id = ?"
  )
  params: list = [model_id]
  if folder:
    norm_folder = folder.strip("/")
    sql += " AND path LIKE ?"
    params.append(f"%{norm_folder}%")
  sql += " ORDER BY path"

  rows = conn.execute(sql, params).fetchall()
  if not rows:
    return root_dir, [], None

  dim = int(rows[0]["dim"])
  matrix = np.empty((len(rows), dim), dtype=np.float32)
  records: list[dict] = []
  for idx, row in enumerate(rows):
    vec = np.frombuffer(row["vector"], dtype="<f4")
    if vec.shape[0] != dim:
      raise ValueError(f"inconsistent vector dim for {row['path']}: {vec.shape[0]} != {dim}")
    matrix[idx] = vec
    records.append(
        {
            "image_id": int(row["image_id"]),
            "rel_path": row["path"],
            "sha256": row["sha256"],
            "size": int(row["size"]),
            "mtime_ns": int(row["mtime_ns"]),
        }
    )
  return root_dir, records, matrix


def rank_vectors(
    query_vec: np.ndarray,
    root_dir: Path,
    records: list[dict],
    matrix: np.ndarray | None,
    limit: int = 10,
    min_score: float = 0.0,
    dedup: bool = False,
    exclude_rel_path: str | None = None,
    force_abs: bool = False,
) -> list[dict]:
  """Rank resident matrix rows by cosine similarity (`matrix @ query_vec`)."""
  if matrix is None or len(records) == 0 or limit <= 0:
    return []

  scores = matrix @ query_vec  # BLAS sgemv over L2-normalized vectors
  order = np.argsort(-scores, kind="stable")

  cwd = Path.cwd().resolve()
  results: list[dict] = []
  seen_hashes: set[str] = set()

  for idx in order:
    score = float(scores[idx])
    if score < min_score:
      break
    rec = records[int(idx)]
    if exclude_rel_path is not None and rec["rel_path"] == exclude_rel_path:
      continue
    if dedup:
      if rec["sha256"] in seen_hashes:
        continue
      seen_hashes.add(rec["sha256"])

    abs_path = (root_dir / rec["rel_path"]).resolve()
    if force_abs:
      display_path = str(abs_path)
    else:
      try:
        display_path = str(abs_path.relative_to(cwd))
      except ValueError:
        display_path = str(abs_path)

    results.append(
        {
            "rank": len(results) + 1,
            "score": round(score, 6),
            "path": display_path,
            "abs_path": str(abs_path),
            "rel_path": rec["rel_path"],
            "sha256": rec["sha256"],
        }
    )
    if len(results) >= limit:
      break

  return results


def search_text(
    query: str,
    db_path: Path,
    client: EmbeddingClient,
    limit: int = 10,
    min_score: float = 0.0,
    folder: str | None = None,
    dedup: bool = False,
    force_abs: bool = False,
) -> list[dict]:
  if not db_path.exists():
    raise FileNotFoundError(
        f"index database not found at {db_path} — run `{Path(sys.argv[0]).name} index` first"
    )
  conn = open_db(db_path, readonly=True)
  try:
    if client.model_id is None:
      client.model_id = get_meta(conn, "model_id")
    if client.model_id is None:
      client.verify()
    model_id = client.model_id
    root_dir, records, matrix = load_resident_index(conn, model_id=model_id, folder=folder)
  finally:
    conn.close()

  query_vec = client.embed_text(query)
  return rank_vectors(
      query_vec=query_vec,
      root_dir=root_dir,
      records=records,
      matrix=matrix,
      limit=limit,
      min_score=min_score,
      dedup=dedup,
      force_abs=force_abs,
  )


def search_similar(
    target_image: Path,
    db_path: Path,
    client: EmbeddingClient,
    limit: int = 10,
    min_score: float = 0.0,
    folder: str | None = None,
    dedup: bool = False,
    include_self: bool = False,
    force_abs: bool = False,
) -> list[dict]:
  if not db_path.exists():
    raise FileNotFoundError(
        f"index database not found at {db_path} — run `{Path(sys.argv[0]).name} index` first"
    )
  target_resolved = target_image.resolve()
  if not target_resolved.is_file():
    raise FileNotFoundError(f"image file not found: {target_image}")

  conn = open_db(db_path, readonly=True)
  try:
    if client.model_id is None:
      client.model_id = get_meta(conn, "model_id")
    if client.model_id is None:
      client.verify()
    model_id = client.model_id
    root_dir, records, matrix = load_resident_index(conn, model_id=model_id, folder=folder)

    exclude_rel: str | None = None
    query_vec: np.ndarray | None = None
    try:
      rel_candidate = target_resolved.relative_to(root_dir.resolve()).as_posix()
      if not include_self:
        exclude_rel = rel_candidate
      row = conn.execute(
          "SELECT vector FROM images WHERE path = ? AND model_id = ? AND vector IS NOT NULL",
          (rel_candidate, model_id),
      ).fetchone()
      if row is not None:
        query_vec = normalize(np.frombuffer(row["vector"], dtype="<f4"))
    except ValueError:
      pass
  finally:
    conn.close()

  if query_vec is None:
    raw = target_resolved.read_bytes()
    query_vec = client.embed_images([(raw, detect_mime(raw, target_resolved))])[0]

  return rank_vectors(
      query_vec=query_vec,
      root_dir=root_dir,
      records=records,
      matrix=matrix,
      limit=limit,
      min_score=min_score,
      dedup=dedup,
      exclude_rel_path=exclude_rel,
      force_abs=force_abs,
  )


def get_status(db_path: Path, client: EmbeddingClient) -> dict:
  probe_info = client.probe()
  info: dict = {
      "db_path": str(db_path),
      "db_exists": db_path.exists(),
      "endpoint": client.base_url,
      "endpoint_reachable": probe_info["reachable"],
      "served_model": probe_info["served_model"],
      "embedding_models": probe_info["embedding_models"],
      "available_models": probe_info["available_models"],
  }
  if not db_path.exists():
    return info

  conn = open_db(db_path, readonly=True)
  try:
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM images WHERE vector IS NOT NULL").fetchone()[0]
    unique_hashes = conn.execute(
        "SELECT COUNT(DISTINCT sha256) FROM images WHERE vector IS NOT NULL"
    ).fetchone()[0]
    errored = conn.execute("SELECT COUNT(*) FROM images WHERE error IS NOT NULL").fetchone()[0]
    dim_row = conn.execute("SELECT dim FROM images WHERE dim IS NOT NULL LIMIT 1").fetchone()
    info.update(
        {
            "root_dir": get_meta(conn, "root_dir"),
            "indexed_model": get_meta(conn, "model_id"),
            "last_indexed_at": get_meta(conn, "last_indexed_at"),
            "db_size_bytes": db_path.stat().st_size,
            "total_rows": int(total),
            "embedded": int(embedded),
            "unique_hashes": int(unique_hashes),
            "errors": int(errored),
            "dim": int(dim_row["dim"]) if dim_row else None,
        }
    )
  finally:
    conn.close()
  return info


def supports_osc8(env: Mapping[str, str] | None = None, isatty: bool | None = None) -> bool:
  """Return True when stdout is an interactive terminal capable of OSC 8 hyperlinks."""
  env = os.environ if env is None else env
  tty = sys.stdout.isatty() if isatty is None else isatty
  return tty and env.get("TERM", "") not in ("", "dumb")


def osc8_link(abs_path: str | Path, label: str) -> str:
  """Wrap `label` in an OSC 8 clickable `file://` hyperlink."""
  uri = Path(abs_path).resolve().as_uri()
  return f"\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\"


def detect_thumb_backend(env: Mapping[str, str] | None = None) -> str:
  """Auto-detect the best inline image protocol (`kitty`, `sixel`, or `ansi`)."""
  env = os.environ if env is None else env
  override = env.get("IMGSEARCH_THUMB", "").strip().lower()
  if override in ("kitty", "sixel", "ansi"):
    return override

  term = env.get("TERM", "").lower()
  term_program = env.get("TERM_PROGRAM", "").lower()

  # Multiplexers swallow graphics escapes unless passthrough is manually configured
  if env.get("TMUX") or term.startswith("screen"):
    return "ansi"

  if (
      env.get("KITTY_WINDOW_ID")
      or "kitty" in term
      or "ghostty" in term
      or term_program in ("kitty", "ghostty", "wezterm")
      or env.get("GHOSTTY_RESOURCES_DIR")
      or env.get("WEZTERM_PANE")
      or env.get("KONSOLE_VERSION")
  ):
    return "kitty"

  if (shutil.which("img2sixel") or shutil.which("magick")) and (
      "sixel" in term
      or term.startswith(("foot", "yaft", "mlterm"))
      or term_program in ("foot", "mintty")
  ):
    return "sixel"

  return "ansi"


def _load_rgb_image(image_path: Path):
  """Load `image_path` with Pillow and composite transparency onto a dark neutral backdrop."""
  try:
    from PIL import Image, ImageOps
  except ImportError as exc:
    raise EmbedError("Pillow is required for -t/--thumb (`pip install pillow`)") from exc

  with Image.open(image_path) as im:
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
      rgba = im.convert("RGBA")
      bg = Image.new("RGBA", rgba.size, (30, 30, 30, 255))
      return Image.alpha_composite(bg, rgba).convert("RGB")
    return im.convert("RGB")


def _render_kitty(img, rows: int, max_cols: int) -> str:
  from PIL import Image

  src_w, src_h = img.size
  thumb = img.copy()
  thumb.thumbnail((max(64, max_cols * 14), max(32, rows * 28)), Image.Resampling.LANCZOS)
  buf = io.BytesIO()
  thumb.save(buf, format="PNG", compress_level=1)
  b64 = base64.b64encode(buf.getvalue()).decode("ascii")

  est_cols = int(round(rows * 2.0 * (src_w / src_h)))
  size_spec = f"c={max_cols}" if est_cols > max_cols else f"r={rows}"

  chunks = [b64[i : i + 4096] for i in range(0, len(b64), 4096)] or [""]
  parts: list[str] = [THUMB_INDENT]
  for idx, chunk in enumerate(chunks):
    more = 1 if idx < len(chunks) - 1 else 0
    if idx == 0:
      parts.append(f"\x1b_Ga=T,f=100,{size_spec},q=2,m={more};{chunk}\x1b\\")
    else:
      parts.append(f"\x1b_Gm={more};{chunk}\x1b\\")
  parts.append("\n")
  return "".join(parts)


def _render_sixel(img, rows: int, max_cols: int) -> str | None:
  from PIL import Image

  thumb = img.copy()
  thumb.thumbnail((max(64, max_cols * 10), max(32, rows * 20)), Image.Resampling.LANCZOS)
  buf = io.BytesIO()
  thumb.save(buf, format="PPM")
  ppm_bytes = buf.getvalue()

  candidates: list[list[str]] = []
  sixel_bin = shutil.which("img2sixel")
  if sixel_bin:
    candidates.append([sixel_bin])
  magick_bin = shutil.which("magick")
  if magick_bin:
    candidates.append([magick_bin, "ppm:-", "sixel:-"])

  for cmd in candidates:
    try:
      proc = subprocess.run(
          cmd,
          input=ppm_bytes,
          stdout=subprocess.PIPE,
          stderr=subprocess.DEVNULL,
          check=True,
          timeout=5,
      )
    except (OSError, subprocess.SubprocessError):
      continue
    if b"\x1bP" in proc.stdout:
      out = proc.stdout.decode("latin1")
      if not out.endswith("\n"):
        out += "\n"
      return THUMB_INDENT + out
  return None


def _render_ansi(img, rows: int, max_cols: int) -> str:
  from PIL import Image

  src_w, src_h = img.size
  ansi_max_cols = min(max_cols, 100)
  h_px = max(2, rows * 2)
  w_px = max(1, int(round(h_px * (src_w / src_h))))
  if w_px > ansi_max_cols:
    w_px = ansi_max_cols
    h_px = max(2, (int(round(w_px * (src_h / src_w))) // 2) * 2)

  resized = img.resize((w_px, h_px), Image.Resampling.LANCZOS)
  arr = np.asarray(resized, dtype=np.uint8)
  lines: list[str] = []
  for y in range(0, h_px, 2):
    top = arr[y].tolist()
    bot = arr[y + 1].tolist()
    cells = [
        f"\x1b[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m▀"
        for (r1, g1, b1), (r2, g2, b2) in zip(top, bot, strict=True)
    ]
    lines.append(THUMB_INDENT + "".join(cells) + "\x1b[0m")
  return "\n".join(lines) + "\n"


def render_thumbnail(
    image_path: Path,
    backend: str = "auto",
    rows: int = DEFAULT_THUMB_ROWS,
    term_cols: int | None = None,
) -> str:
  """Render an inline terminal thumbnail string for `image_path`."""
  resolved = detect_thumb_backend() if backend == "auto" else backend
  if resolved not in ("kitty", "sixel", "ansi"):
    raise ValueError(f"unknown thumbnail backend: {backend!r}")

  rows = max(1, rows)
  cols = term_cols if term_cols is not None else shutil.get_terminal_size((80, 24)).columns
  max_cols = max(16, cols - len(THUMB_INDENT) - 2)

  img = _load_rgb_image(image_path)
  if img.size[0] < 1 or img.size[1] < 1:
    return ""

  if resolved == "kitty":
    return _render_kitty(img, rows=rows, max_cols=max_cols)
  if resolved == "sixel":
    sixel_out = _render_sixel(img, rows=rows, max_cols=max_cols)
    if sixel_out is not None:
      return sixel_out
  return _render_ansi(img, rows=rows, max_cols=max_cols)


def _resolve_thumb_mode(args: argparse.Namespace) -> str | None:
  backend = getattr(args, "thumb_backend", None)
  if backend is not None:
    return backend
  if getattr(args, "thumb", False):
    return "auto"
  return None


def print_results(results: list[dict], args: argparse.Namespace) -> int:
  if args.json:
    print(json.dumps(results, indent=2))
    return 0 if results else 1
  if args.null:
    for item in results:
      sys.stdout.write(item["path"] + "\0")
    sys.stdout.flush()
    return 0 if results else 1
  if args.paths_only:
    for item in results:
      print(item["path"])
    return 0 if results else 1

  if not results:
    print("No matching images found.", file=sys.stderr)
    return 1

  use_osc8 = supports_osc8()
  thumb_mode = _resolve_thumb_mode(args)
  backend = (
      (detect_thumb_backend() if thumb_mode == "auto" else thumb_mode)
      if thumb_mode is not None
      else None
  )
  term_cols = shutil.get_terminal_size((80, 24)).columns if backend else None
  thumb_rows = max(1, getattr(args, "thumb_rows", DEFAULT_THUMB_ROWS))

  for idx, item in enumerate(results):
    label = osc8_link(item["abs_path"], item["path"]) if use_osc8 else item["path"]
    print(f"{item['rank']:>2}.  {item['score']:.4f}  {label}")
    if backend is not None:
      try:
        rendered = render_thumbnail(
            Path(item["abs_path"]),
            backend=backend,
            rows=thumb_rows,
            term_cols=term_cols,
        )
      except OSError as exc:
        print(f"{THUMB_INDENT}(thumbnail unavailable: {exc})", file=sys.stderr)
      else:
        sys.stdout.write(rendered)
        sys.stdout.flush()
      if idx < len(results) - 1:
        print()
  return 0


def _add_common_flags(p: argparse.ArgumentParser) -> None:
  p.add_argument(
      "--db", type=Path, default=DEFAULT_DB, help=f"SQLite index path (default: {DEFAULT_DB})"
  )
  p.add_argument(
      "--endpoint",
      default=DEFAULT_ENDPOINT,
      help=f"Embedding server URL (default: {DEFAULT_ENDPOINT})",
  )
  model_help = (
      f"Model ID substring (default: {DEFAULT_MODEL})"
      if DEFAULT_MODEL
      else "Model ID substring (default: first embedding model reported by endpoint)"
  )
  p.add_argument("--model", default=DEFAULT_MODEL, help=model_help)


def _add_search_output_flags(p: argparse.ArgumentParser) -> None:
  p.add_argument("-n", "--limit", type=int, default=10, help="Max results to return (default: 10)")
  p.add_argument("--min-score", type=float, default=0.0, help="Minimum cosine similarity threshold")
  p.add_argument("-f", "--folder", default=None, help="Substring filter on indexed relative path")
  p.add_argument("--dedup", action="store_true", help="Collapse byte-identical duplicate images")
  p.add_argument("--abs", action="store_true", help="Always output absolute file paths")
  p.add_argument(
      "-t",
      "--thumb",
      action="store_true",
      help="Display inline thumbnails in the terminal (use --thumb=kitty|sixel|ansi to override)",
  )
  p.add_argument(
      "--thumb-backend",
      choices=THUMB_BACKENDS,
      default=None,
      help="Thumbnail protocol override (default: auto)",
  )
  p.add_argument(
      "--thumb-rows",
      type=int,
      default=DEFAULT_THUMB_ROWS,
      help=f"Thumbnail height in terminal rows (default: {DEFAULT_THUMB_ROWS})",
  )
  p.add_argument("-l", "--paths-only", action="store_true", help="Print matching file paths only")
  p.add_argument(
      "-0", "--null", action="store_true", help="Print NUL-delimited paths (for xargs -0)"
  )
  p.add_argument("--json", action="store_true", help="Output results as JSON")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="imgsearch",
      description=(
          "Build and search a semantic vector index over local images using EmbeddingGemma 2.\n\n"
          "Shorthand: `imgsearch <query>` runs `imgsearch search <query>`."
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  sub = parser.add_subparsers(dest="command", required=True)

  p_index = sub.add_parser("index", help="Scan directory and build/update the image vector index")
  p_index.add_argument(
      "root",
      nargs="?",
      type=Path,
      default=DEFAULT_ROOT,
      help=f"Root directory to scan for images (default: {DEFAULT_ROOT})",
  )
  p_index.add_argument(
      "-b",
      "--batch-size",
      type=int,
      default=DEFAULT_BATCH_SIZE,
      help=f"Images per embedding request (default: {DEFAULT_BATCH_SIZE})",
  )
  p_index.add_argument("--force", action="store_true", help="Re-embed all images from scratch")
  p_index.add_argument("--retry-errors", action="store_true", help="Retry previously failed files")
  p_index.add_argument(
      "--keep-missing", action="store_true", help="Do not prune deleted files from DB"
  )
  p_index.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
  p_index.add_argument("--json", action="store_true", help="Print summary as JSON")
  _add_common_flags(p_index)

  p_search = sub.add_parser("search", help="Search indexed images with a natural-language query")
  p_search.add_argument("query", nargs="+", help="Natural-language search query")
  _add_search_output_flags(p_search)
  _add_common_flags(p_search)

  p_sim = sub.add_parser("similar", help="Find indexed images visually similar to a given image")
  p_sim.add_argument("image", type=Path, help="Path to query image (indexed or external)")
  p_sim.add_argument(
      "--include-self", action="store_true", help="Include the query image itself in results"
  )
  _add_search_output_flags(p_sim)
  _add_common_flags(p_sim)

  p_status = sub.add_parser("status", help="Show index statistics and endpoint health")
  p_status.add_argument("--json", action="store_true", help="Output status as JSON")
  _add_common_flags(p_status)

  return parser


def _normalize_argv(raw_args: list[str]) -> list[str]:
  """Expand `--thumb=<mode>` / `-t=<mode>` and apply search subcommand shorthand."""
  expanded: list[str] = []
  for token in raw_args:
    if token.startswith("--thumb="):
      expanded.extend(["--thumb", "--thumb-backend", token.split("=", 1)[1]])
    elif token.startswith("-t="):
      expanded.extend(["-t", "--thumb-backend", token.split("=", 1)[1]])
    else:
      expanded.append(token)

  if expanded and expanded[0] not in SUBCOMMANDS and expanded[0] not in ("-h", "--help"):
    expanded = ["search", *expanded]
  return expanded


def main(argv: list[str] | None = None) -> int:
  raw_args = _normalize_argv(list(sys.argv[1:] if argv is None else argv))

  parser = build_parser()
  args = parser.parse_args(raw_args)

  try:
    with EmbeddingClient(base_url=args.endpoint, model_id=args.model) as client:
      if args.command == "index":
        if args.batch_size < 1:
          parser.error("--batch-size must be >= 1")
        summary = build_index(
            root=args.root,
            db_path=args.db,
            client=client,
            batch_size=args.batch_size,
            force=args.force,
            retry_errors=args.retry_errors,
            keep_missing=args.keep_missing,
            quiet=args.quiet,
        )
        if args.json:
          print(json.dumps(summary, indent=2))
        elif not args.quiet:
          print(
              f"Indexed: {summary['embedded']} new, {summary['up_to_date']} unchanged, "
              f"{summary['reused_hash']} hash-reused, {summary['failed']} failed "
              f"in {format_duration(summary['elapsed_seconds'])} -> {args.db}"
          )
        return 0 if summary["failed"] == 0 else 2

      if args.command == "search":
        query_text = " ".join(args.query).strip()
        if not query_text:
          parser.error("search query must not be empty")
        results = search_text(
            query=query_text,
            db_path=args.db,
            client=client,
            limit=args.limit,
            min_score=args.min_score,
            folder=args.folder,
            dedup=args.dedup,
            force_abs=args.abs,
        )
        return print_results(results, args)

      if args.command == "similar":
        results = search_similar(
            target_image=args.image,
            db_path=args.db,
            client=client,
            limit=args.limit,
            min_score=args.min_score,
            folder=args.folder,
            dedup=args.dedup,
            include_self=args.include_self,
            force_abs=args.abs,
        )
        return print_results(results, args)

      if args.command == "status":
        info = get_status(db_path=args.db, client=client)
        if args.json:
          print(json.dumps(info, indent=2))
        else:
          print(
              f"Database:       {info['db_path']} ({'exists' if info['db_exists'] else 'missing'})"
          )
          if info["db_exists"]:
            size_mb = info["db_size_bytes"] / (1024 * 1024)
            print(f"Root directory: {info['root_dir']}")
            print(f"Indexed model:  {info['indexed_model']} (dim={info['dim']})")
            print(
                f"Images:         {info['embedded']}/{info['total_rows']} embedded "
                f"({info['unique_hashes']} unique hashes, {info['errors']} errors)"
            )
            print(f"DB size:        {size_mb:.2f} MB")
          if not info["endpoint_reachable"]:
            endpoint_state = "UNREACHABLE"
          elif info["served_model"]:
            endpoint_state = f"reachable (embedding model: {info['served_model']!r})"
          else:
            endpoint_state = (
                f"reachable (no embedding model detected; available: {info['available_models']})"
            )
          print(f"Endpoint:       {info['endpoint']} — {endpoint_state}")
        return 0
  except BrokenPipeError:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    return 0
  except (EmbedError, FileNotFoundError, ValueError) as exc:
    print(f"imgsearch: {exc}", file=sys.stderr)
    return 2

  return 0


if __name__ == "__main__":
  sys.exit(main())
