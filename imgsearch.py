#!/usr/bin/env python3
"""Vector indexer and search CLI for images using EmbeddingGemma 2."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

from embed_base import (
    EmbedError,
    EmbeddingBackend,
    EndpointUnreachable,  # noqa: F401 - re-exported for callers
    IMAGE_EXTENSIONS,
    ItemRejected,
    ModelMismatch,
    QUERY_PREFIX,  # noqa: F401 - re-exported for callers
    decode_rgb_image,
    detect_mime,
    models_compatible,
    normalize,
    resolve_local_hf_model,
)
from local_embed import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_DIR,
    LocalEmbeddingClient,
    has_local_weights,
)
from remote_embed import DEFAULT_ENDPOINT, DEFAULT_MODEL, EmbeddingClient

DEFAULT_ROOT = Path(".")
DEFAULT_DB = Path("index.db")
DEFAULT_BATCH_SIZE = 8
BUSY_TIMEOUT_MS = 30_000

SUBCOMMANDS = {"index", "search", "similar", "status"}
THUMB_BACKENDS = ("auto", "kitty", "sixel", "ansi")
DEFAULT_THUMB_ROWS = 10
MAX_THUMB_ROWS = 200
THUMB_INDENT = "    "

VALUE_FLAGS = {
    "--db",
    "--endpoint",
    "--model",
    "--model-dir",
    "--device",
    "-n",
    "--limit",
    "--min-score",
    "-f",
    "--folder",
    "--thumb-backend",
    "--thumb-rows",
    "-b",
    "--batch-size",
}
SHORT_VALUE_PREFIXES = ("-n", "-f", "-b")

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
  """Open SQLite index in read-write (`DELETE` journal) or read-only mode."""
  db_path = Path(db_path)
  timeout_sec = BUSY_TIMEOUT_MS / 1000.0
  if readonly:
    uri = db_path.resolve().as_uri()
    conn: sqlite3.Connection | None = None
    try:
      conn = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=timeout_sec)
      conn.row_factory = sqlite3.Row
      conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
      conn.execute("PRAGMA schema_version").fetchone()
      return conn
    except sqlite3.OperationalError as exc:
      if conn is not None:
        conn.close()
      msg = str(exc).lower()
      can_fallback = ("readonly" in msg or "unable to open" in msg) and not any(
          tok in msg for tok in ("locked", "busy")
      )
      if not can_fallback:
        raise
      # Fallback for legacy WAL DBs in non-writable directories.
      conn = sqlite3.connect(
          f"{uri}?mode=ro&immutable=1", uri=True, timeout=timeout_sec
      )
      conn.row_factory = sqlite3.Row
      return conn

  db_path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(db_path, timeout=timeout_sec)
  try:
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)
  except Exception:
    conn.close()
    raise
  return conn


def get_meta(
    conn: sqlite3.Connection, key: str, default: str | None = None
) -> str | None:
  row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
  return row["value"] if row is not None else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
  conn.execute(
      "INSERT INTO meta(key, value) VALUES(?, ?) "
      "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      (key, value),
  )


def _safe_resolve(path: Path) -> Path | None:
  try:
    return path.resolve()
  except (OSError, RuntimeError):
    return None


def discover_images(
    root: Path,
    db_path: Path,
    exclude_dirs: list[Path] | None = None,
) -> list[Path]:
  """Recursively find image files under `root`, skipping hidden/weight dirs."""
  db_resolved = _safe_resolve(db_path)
  excluded = {
      res for d in (exclude_dirs or []) if (res := _safe_resolve(Path(d)))
  }
  found: list[Path] = []
  for dirpath, dirnames, filenames in os.walk(root):
    cur = Path(dirpath)
    pruned_dirs: list[str] = []
    for d in sorted(dirnames):
      if d.startswith("."):
        continue
      sub = cur / d
      sub_resolved = _safe_resolve(sub)
      if sub_resolved is None or sub_resolved in excluded:
        continue
      if (
          has_local_weights(sub)
          or (sub / "sentence_bert_config.json").is_file()
      ):
        continue
      pruned_dirs.append(d)
    dirnames[:] = pruned_dirs

    for fname in sorted(filenames):
      if fname.startswith("."):
        continue
      p = cur / fname
      if p.suffix.lower() not in IMAGE_EXTENSIONS:
        continue
      p_resolved = _safe_resolve(p)
      if p_resolved is not None and p_resolved == db_resolved:
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
    client: EmbeddingBackend,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    retry_errors: bool = False,
    keep_missing: bool = False,
    quiet: bool = False,
) -> dict:
  """Scan `root`, reuse unchanged/duplicate vectors, and embed new images."""
  root = root.resolve()
  if not root.is_dir():
    raise FileNotFoundError(f"root directory does not exist: {root}")

  served_model = client.verify()
  conn = open_db(db_path)
  try:
    existing_rows = {
        row["path"]: row for row in conn.execute("SELECT * FROM images")
    }
    prev_root = get_meta(conn, "root_dir")
    if (
        not force
        and prev_root
        and existing_rows
        and Path(prev_root).resolve() != root
    ):
      raise EmbedError(
          f"database {db_path} already indexes {prev_root}; "
          "pass --db for a separate index or --force to re-root"
      )

    existing_model = get_meta(conn, "model_id")
    if (
        not force
        and existing_model
        and models_compatible(existing_model, client.model_id)
    ):
      client.model_id = existing_model
    endpoint_label = getattr(client, "base_url", "local")
    if not existing_rows or client.model_id == existing_model:
      with conn:
        set_meta(conn, "root_dir", str(root))
        set_meta(conn, "model_id", str(client.model_id))
        set_meta(conn, "served_model", served_model)
        set_meta(conn, "endpoint", endpoint_label)

    exclude_dirs: list[Path] = []
    wdir = getattr(client, "weights_dir", None)
    if wdir is not None:
      exclude_dirs.append(Path(wdir))
    files = discover_images(root, db_path, exclude_dirs=exclude_dirs)

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
    to_embed: list[tuple[str, Path]] = []

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
          recovered = (
              not prev["sha256"] or "Permission denied" in prev["error"]
          ) and os.access(full_path, os.R_OK)
          if not recovered:
            skipped_errors += 1
            continue

      if not force and hash_vectors:
        try:
          raw_scan = full_path.read_bytes()
          st_scan = full_path.stat()
        except OSError as exc:
          _record_error(
              conn,
              rel_path,
              "",
              st.st_size,
              st.st_mtime_ns,
              client.model_id,
              str(exc),
          )
          failed += 1
          continue
        digest_scan = sha256_bytes(raw_scan)
        if digest_scan in hash_vectors:
          dim, vec_bytes = hash_vectors[digest_scan]
          _record_vector(
              conn,
              rel_path,
              digest_scan,
              st_scan.st_size,
              st_scan.st_mtime_ns,
              str(client.model_id),
              dim,
              vec_bytes,
          )
          reused_hash += 1
          continue
      elif not os.access(full_path, os.R_OK):
        try:
          full_path.read_bytes()
        except OSError as exc:
          _record_error(
              conn,
              rel_path,
              "",
              st.st_size,
              st.st_mtime_ns,
              client.model_id,
              str(exc),
          )
          failed += 1
          continue

      to_embed.append((rel_path, full_path))

    pruned = 0
    if not keep_missing:
      missing = set(existing_rows.keys()) - seen_rel_paths
      if missing:
        with conn:
          conn.executemany(
              "DELETE FROM images WHERE path = ?", [(p,) for p in missing]
          )
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
          f"(up-to-date: {up_to_date}, hash-reused: {reused_hash}, "
          f"to-embed: {total_todo}{extra_str})",
          file=sys.stderr,
      )

    embedded = 0
    t_start = time.perf_counter()

    for start in range(0, total_todo, batch_size):
      chunk = to_embed[start : start + batch_size]
      loaded: list[tuple[tuple[str, int, int, str], bytes, str]] = []
      for rel_path, full_path in chunk:
        try:
          raw = full_path.read_bytes()
          st = full_path.stat()
        except OSError as exc:
          _record_error(conn, rel_path, "", 0, 0, client.model_id, str(exc))
          failed += 1
          continue
        digest = sha256_bytes(raw)
        if not force and digest in hash_vectors:
          dim, vec_bytes = hash_vectors[digest]
          _record_vector(
              conn,
              rel_path,
              digest,
              st.st_size,
              st.st_mtime_ns,
              str(client.model_id),
              dim,
              vec_bytes,
          )
          reused_hash += 1
          continue
        meta_entry = (rel_path, st.st_size, st.st_mtime_ns, digest)
        loaded.append((meta_entry, raw, detect_mime(raw, full_path)))

      if loaded:
        try:
          vectors = client.embed_images(
              [(raw, mime) for _, raw, mime in loaded]
          )
          now_ts = time.time()
          with conn:
            for (rel_path, size, mtime_ns, digest), vec in zip(
                (item[0] for item in loaded), vectors, strict=True
            ):
              vec_f32 = np.asarray(vec, dtype="<f4")
              vec_bytes = vec_f32.tobytes()
              dim = int(vec_f32.shape[0])
              hash_vectors[digest] = (dim, vec_bytes)
              _upsert_vector(
                  conn,
                  rel_path,
                  digest,
                  size,
                  mtime_ns,
                  str(client.model_id),
                  dim,
                  vec_bytes,
                  now_ts,
              )
              embedded += 1
        except ItemRejected:
          for (rel_path, size, mtime_ns, digest), raw, mime in loaded:
            try:
              vec = client.embed_images([(raw, mime)])[0]
              vec_f32 = np.asarray(vec, dtype="<f4")
              vec_bytes = vec_f32.tobytes()
              dim = int(vec_f32.shape[0])
              hash_vectors[digest] = (dim, vec_bytes)
              _record_vector(
                  conn,
                  rel_path,
                  digest,
                  size,
                  mtime_ns,
                  str(client.model_id),
                  dim,
                  vec_bytes,
              )
              embedded += 1
            except ItemRejected as exc:
              _record_error(
                  conn,
                  rel_path,
                  digest,
                  size,
                  mtime_ns,
                  client.model_id,
                  str(exc),
              )
              failed += 1

      if not quiet and total_todo > 0:
        done = min(start + len(chunk), total_todo)
        elapsed = max(time.perf_counter() - t_start, 1e-6)
        rate = done / elapsed
        eta = (total_todo - done) / rate if rate > 0 else 0.0
        pct = 100.0 * done / total_todo
        msg = (
            f"\r[{done}/{total_todo}] {pct:5.1f}% | "
            f"{rate:5.1f} img/s | elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta)}   "
        )
        sys.stderr.write(msg)
        sys.stderr.flush()

    if not quiet and total_todo > 0:
      sys.stderr.write("\n")
      sys.stderr.flush()

    elapsed_total = time.perf_counter() - t_start
    with conn:
      set_meta(conn, "root_dir", str(root))
      set_meta(conn, "model_id", str(client.model_id))
      set_meta(conn, "served_model", served_model)
      set_meta(conn, "endpoint", endpoint_label)
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


def _upsert_vector(
    conn: sqlite3.Connection,
    rel_path: str,
    digest: str,
    size: int,
    mtime_ns: int,
    model_id: str,
    dim: int,
    vec_bytes: bytes,
    now_ts: float,
) -> None:
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
      (rel_path, digest, size, mtime_ns, model_id, dim, vec_bytes, now_ts),
  )


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
    _upsert_vector(
        conn,
        rel_path,
        digest,
        size,
        mtime_ns,
        model_id,
        dim,
        vec_bytes,
        time.time(),
    )


def _record_error(
    conn: sqlite3.Connection,
    rel_path: str,
    digest: str,
    size: int,
    mtime_ns: int,
    model_id: str | None,
    error: str,
) -> None:
  with conn:
    conn.execute(
        """
        INSERT INTO images(
            path, sha256, size, mtime_ns, model_id, dim,
            vector, embedded_at, error
        )
        VALUES(?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
        ON CONFLICT(path) DO UPDATE SET
            sha256=excluded.sha256,
            size=excluded.size,
            mtime_ns=excluded.mtime_ns,
            model_id=excluded.model_id,
            dim=NULL,
            vector=NULL,
            embedded_at=NULL,
            error=excluded.error
        """,
        (rel_path, digest, size, mtime_ns, model_id, error),
    )


def _escape_like(text: str) -> str:
  return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def load_resident_index(
    conn: sqlite3.Connection,
    model_id: str,
    folder: str | None = None,
) -> tuple[Path, list[dict], np.ndarray | None]:
  """Load float32 vectors from SQLite into a contiguous `(N, dim)` matrix."""
  root_str = get_meta(conn, "root_dir")
  root_dir = Path(root_str) if root_str else DEFAULT_ROOT

  sql = (
      "SELECT image_id, path, sha256, size, mtime_ns, dim, vector "
      "FROM images WHERE vector IS NOT NULL AND model_id = ?"
  )
  params: list = [model_id]
  if folder:
    norm_folder = folder.replace("\\", "/").strip("/")
    if norm_folder:
      sql += " AND path LIKE ? ESCAPE '\\'"
      params.append(f"%{_escape_like(norm_folder)}%")
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
      raise ValueError(
          f"inconsistent vector dim for {row['path']}: {vec.shape[0]} != {dim}"
      )
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
  if matrix.shape[1] != query_vec.shape[0]:
    raise ModelMismatch(
        f"query vector dim ({query_vec.shape[0]}) does not match "
        f"indexed dim ({matrix.shape[1]})"
    )

  scores = matrix @ query_vec
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

    abs_path = root_dir.resolve() / rec["rel_path"]
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


def _align_backend_to_db(
    db_model: str | None,
    client: EmbeddingBackend,
    verify_remote: bool = True,
) -> str:
  """Align `client` with `db_model` and verify vector-space compatibility."""
  if getattr(client, "is_local", False):
    if not client._explicit_model and db_model:
      client.hf_model_id = resolve_local_hf_model(db_model)
      client.served_model = client.hf_model_id
      client.model_id = db_model
    elif db_model and not models_compatible(db_model, client.hf_model_id):
      raise ModelMismatch(
          f"index uses {db_model!r}, which does not match "
          f"local model {client.hf_model_id!r}"
      )
    return db_model or client.model_id or resolve_local_hf_model(None)

  if client.model_id is None and db_model:
    client.model_id = db_model
  if verify_remote or client.model_id is None:
    client.verify()
  if db_model and not models_compatible(
      db_model, client.served_model or client.model_id
  ):
    raise ModelMismatch(
        f"index uses {db_model!r}, which does not match "
        f"endpoint model {(client.served_model or client.model_id)!r}"
    )
  return db_model or str(client.model_id)


def search_text(
    query: str,
    db_path: Path,
    client: EmbeddingBackend,
    limit: int = 10,
    min_score: float = 0.0,
    folder: str | None = None,
    dedup: bool = False,
    force_abs: bool = False,
) -> list[dict]:
  if not db_path.exists():
    prog = Path(sys.argv[0]).name
    raise FileNotFoundError(
        f"index database not found at {db_path} — run `{prog} index` first"
    )
  conn = open_db(db_path, readonly=True)
  try:
    db_model = get_meta(conn, "model_id")
    model_id = _align_backend_to_db(db_model, client, verify_remote=True)
    root_dir, records, matrix = load_resident_index(
        conn, model_id=model_id, folder=folder
    )
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


def _relative_to_root(target: Path, root_dir: Path) -> str | None:
  """Resolve `target` relative to `root_dir`, preserving intra-root symlinks."""
  root_resolved = root_dir.resolve()
  lexical = target.parent.resolve() / target.name
  for candidate in (lexical, target.resolve()):
    try:
      return candidate.relative_to(root_resolved).as_posix()
    except ValueError:
      continue
  return None


def search_similar(
    target_image: Path,
    db_path: Path,
    client: EmbeddingBackend,
    limit: int = 10,
    min_score: float = 0.0,
    folder: str | None = None,
    dedup: bool = False,
    include_self: bool = False,
    force_abs: bool = False,
) -> list[dict]:
  if not db_path.exists():
    prog = Path(sys.argv[0]).name
    raise FileNotFoundError(
        f"index database not found at {db_path} — run `{prog} index` first"
    )
  if not target_image.is_file():
    raise FileNotFoundError(f"image file not found: {target_image}")
  target_stat = target_image.stat()

  conn = open_db(db_path, readonly=True)
  try:
    db_model = get_meta(conn, "model_id")
    # Defer remote HTTP verification unless we miss the SQLite vector cache
    # or the user explicitly passed `--model`.
    explicit_remote = (
        not getattr(client, "is_local", False) and client.model_id is not None
    )
    model_id = _align_backend_to_db(
        db_model, client, verify_remote=explicit_remote
    )
    root_dir, records, matrix = load_resident_index(
        conn, model_id=model_id, folder=folder
    )

    exclude_rel: str | None = None
    query_vec: np.ndarray | None = None
    rel_candidate = _relative_to_root(target_image, root_dir)
    if rel_candidate is not None:
      if not include_self:
        exclude_rel = rel_candidate
      row = conn.execute(
          "SELECT size, mtime_ns, vector FROM images "
          "WHERE path = ? AND model_id = ? AND vector IS NOT NULL",
          (rel_candidate, model_id),
      ).fetchone()
      if (
          row is not None
          and row["size"] == target_stat.st_size
          and row["mtime_ns"] == target_stat.st_mtime_ns
      ):
        query_vec = normalize(np.frombuffer(row["vector"], dtype="<f4"))
  finally:
    conn.close()

  if query_vec is None:
    if not getattr(client, "is_local", False):
      _align_backend_to_db(db_model, client, verify_remote=True)
    raw = target_image.read_bytes()
    mime = detect_mime(raw, target_image)
    query_vec = client.embed_images([(raw, mime)])[0]

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


def get_status(db_path: Path, client: EmbeddingBackend) -> dict:
  probe_info = client.probe()
  hf_id = getattr(client, "hf_model_id", "")
  endpoint_desc = getattr(client, "base_url", f"local ({hf_id})")
  info: dict = {
      "db_path": str(db_path),
      "db_exists": db_path.exists(),
      "endpoint": endpoint_desc,
      "endpoint_reachable": probe_info["reachable"],
      "model_matches": probe_info.get("matches", False),
      "served_model": probe_info["served_model"],
      "embedding_models": probe_info["embedding_models"],
      "available_models": probe_info["available_models"],
  }
  if "weights_dir" in probe_info:
    info["weights_dir"] = probe_info["weights_dir"]
    info["weights_downloaded"] = probe_info["weights_downloaded"]
  if not db_path.exists():
    return info

  conn = open_db(db_path, readonly=True)
  try:
    indexed_model = get_meta(conn, "model_id")
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if indexed_model:
      embedded = conn.execute(
          "SELECT COUNT(*) FROM images "
          "WHERE vector IS NOT NULL AND model_id = ?",
          (indexed_model,),
      ).fetchone()[0]
      unique_hashes = conn.execute(
          "SELECT COUNT(DISTINCT sha256) FROM images "
          "WHERE vector IS NOT NULL AND model_id = ?",
          (indexed_model,),
      ).fetchone()[0]
      dim_row = conn.execute(
          "SELECT dim FROM images "
          "WHERE vector IS NOT NULL AND model_id = ? LIMIT 1",
          (indexed_model,),
      ).fetchone()
    else:
      embedded = conn.execute(
          "SELECT COUNT(*) FROM images WHERE vector IS NOT NULL"
      ).fetchone()[0]
      unique_hashes = conn.execute(
          "SELECT COUNT(DISTINCT sha256) FROM images WHERE vector IS NOT NULL"
      ).fetchone()[0]
      dim_row = conn.execute(
          "SELECT dim FROM images WHERE vector IS NOT NULL LIMIT 1"
      ).fetchone()
    errored = conn.execute(
        "SELECT COUNT(*) FROM images WHERE error IS NOT NULL"
    ).fetchone()[0]
    info.update(
        {
            "root_dir": get_meta(conn, "root_dir"),
            "indexed_model": indexed_model,
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


def sanitize_tty_text(text: str) -> str:
  """Replace C0, DEL, and C1 control chars with U+FFFD for safe TTY output."""
  return "".join(
      "\ufffd" if (ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F) else ch
      for ch in text
  )


def supports_osc8(
    env: Mapping[str, str] | None = None, isatty: bool | None = None
) -> bool:
  """Return True when stdout is an interactive TTY supporting OSC 8 links."""
  env = os.environ if env is None else env
  tty = sys.stdout.isatty() if isatty is None else isatty
  return tty and env.get("TERM", "") not in ("", "dumb")


def osc8_link(abs_path: str | Path, label: str) -> str:
  """Wrap sanitized `label` in an OSC 8 clickable `file://` hyperlink."""
  uri = Path(abs_path).resolve().as_uri()
  safe_label = sanitize_tty_text(label)
  return f"\x1b]8;;{uri}\x1b\\{safe_label}\x1b]8;;\x1b\\"


def detect_thumb_backend(env: Mapping[str, str] | None = None) -> str:
  """Auto-detect the best inline image protocol (`kitty`, `sixel`, `ansi`)."""
  env = os.environ if env is None else env
  override = env.get("IMGSEARCH_THUMB", "").strip().lower()
  if override in ("kitty", "sixel", "ansi"):
    return override

  term = env.get("TERM", "").lower()
  term_program = env.get("TERM_PROGRAM", "").lower()

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


def _render_kitty(img, rows: int, max_cols: int) -> str:
  from PIL import Image

  src_w, src_h = img.size
  thumb = img.copy()
  thumb.thumbnail(
      (max(64, max_cols * 14), max(32, rows * 28)), Image.Resampling.LANCZOS
  )
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

  src_w, src_h = img.size
  target_h = max(32, rows * 18)
  target_w = max(1, int(round(target_h * (src_w / src_h))))
  max_px_w = max(64, max_cols * 9)
  if target_w > max_px_w:
    target_w = max_px_w
    target_h = max(1, int(round(target_w * (src_h / src_w))))
  thumb = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
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
  clamped_rows = min(rows, MAX_THUMB_ROWS)
  h_px = max(2, clamped_rows * 2)
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

  rows = max(1, min(rows, MAX_THUMB_ROWS))
  cols = (
      term_cols
      if term_cols is not None
      else shutil.get_terminal_size((80, 24)).columns
  )
  max_cols = max(16, cols - len(THUMB_INDENT) - 2)

  img = decode_rgb_image(image_path)
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
    tty = sys.stdout.isatty()
    for item in results:
      out_path = sanitize_tty_text(item["path"]) if tty else item["path"]
      print(out_path)
    return 0 if results else 1

  if not results:
    print("No matching images found.", file=sys.stderr)
    return 1

  tty = sys.stdout.isatty()
  use_osc8 = supports_osc8(isatty=tty)
  thumb_mode = _resolve_thumb_mode(args)
  backend = (
      (detect_thumb_backend() if thumb_mode == "auto" else thumb_mode)
      if thumb_mode is not None
      else None
  )
  term_cols = shutil.get_terminal_size((80, 24)).columns if backend else None
  thumb_rows = max(1, getattr(args, "thumb_rows", DEFAULT_THUMB_ROWS))

  for idx, item in enumerate(results):
    if use_osc8:
      label = osc8_link(item["abs_path"], item["path"])
    elif tty:
      label = sanitize_tty_text(item["path"])
    else:
      label = item["path"]
    print(f"{item['rank']:>2}.  {item['score']:.4f}  {label}")
    if backend is not None:
      try:
        rendered = render_thumbnail(
            Path(item["abs_path"]),
            backend=backend,
            rows=thumb_rows,
            term_cols=term_cols,
        )
      except (OSError, ValueError, SyntaxError, EOFError, EmbedError) as exc:
        print(f"{THUMB_INDENT}(thumbnail unavailable: {exc})", file=sys.stderr)
      else:
        sys.stdout.write(rendered)
        sys.stdout.flush()
      if idx < len(results) - 1:
        print()
  return 0


def _add_common_flags(p: argparse.ArgumentParser) -> None:
  p.add_argument(
      "--db",
      type=Path,
      default=DEFAULT_DB,
      help=f"SQLite index path (default: {DEFAULT_DB})",
  )
  endpoint_help = (
      f"Embedding server URL (default: {DEFAULT_ENDPOINT})"
      if DEFAULT_ENDPOINT
      else "Embedding server URL (if omitted, runs model locally in-process)"
  )
  p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=endpoint_help)
  model_help = (
      f"Model ID substring or HF repo (default: {DEFAULT_MODEL})"
      if DEFAULT_MODEL
      else "Model ID substring or HF repo (default: auto-detected)"
  )
  p.add_argument("--model", default=DEFAULT_MODEL, help=model_help)
  model_dir_help = (
      f"Directory for local model weights (default: {DEFAULT_MODEL_DIR})"
      if DEFAULT_MODEL_DIR
      else "Directory for local model weights (default: ./models/<model>)"
  )
  p.add_argument(
      "--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=model_dir_help
  )
  p.add_argument(
      "--device",
      default=DEFAULT_DEVICE,
      help="Torch device when running locally (e.g. cpu, cuda; default: auto)",
  )


def _add_search_output_flags(p: argparse.ArgumentParser) -> None:
  p.add_argument(
      "-n",
      "--limit",
      type=int,
      default=10,
      help="Max results to return (default: 10)",
  )
  p.add_argument(
      "--min-score",
      type=float,
      default=0.0,
      help="Minimum cosine similarity threshold (-1.0 to 1.0)",
  )
  p.add_argument(
      "-f",
      "--folder",
      default=None,
      help="Substring filter on indexed relative path",
  )
  p.add_argument(
      "--dedup",
      action="store_true",
      help="Collapse byte-identical duplicate images",
  )
  p.add_argument(
      "--abs",
      action="store_true",
      help="Always output absolute file paths",
  )
  p.add_argument(
      "-t",
      "--thumb",
      action="store_true",
      help="Display inline thumbnails (use --thumb=kitty|sixel|ansi)",
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
  p.add_argument(
      "-l",
      "--paths-only",
      action="store_true",
      help="Print matching file paths only",
  )
  p.add_argument(
      "-0",
      "--null",
      action="store_true",
      help="Print NUL-delimited paths (for xargs -0)",
  )
  p.add_argument("--json", action="store_true", help="Output results as JSON")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="imgsearch",
      description=(
          "Build and search a semantic vector index over local images.\n\n"
          "Shorthand: `imgsearch <query>` runs `imgsearch search <query>`."
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  sub = parser.add_subparsers(dest="command", required=True)

  p_index = sub.add_parser(
      "index", help="Scan directory and build/update the image vector index"
  )
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
  p_index.add_argument(
      "--force", action="store_true", help="Re-embed all images from scratch"
  )
  p_index.add_argument(
      "--retry-errors",
      action="store_true",
      help="Retry previously failed files",
  )
  p_index.add_argument(
      "--keep-missing",
      action="store_true",
      help="Do not prune deleted files from DB",
  )
  p_index.add_argument(
      "-q", "--quiet", action="store_true", help="Suppress progress output"
  )
  p_index.add_argument(
      "--json", action="store_true", help="Print summary as JSON"
  )
  _add_common_flags(p_index)

  p_search = sub.add_parser(
      "search", help="Search indexed images with a natural-language query"
  )
  p_search.add_argument("query", nargs="+", help="Natural-language query")
  _add_search_output_flags(p_search)
  _add_common_flags(p_search)

  p_sim = sub.add_parser(
      "similar", help="Find indexed images visually similar to a given image"
  )
  p_sim.add_argument("image", type=Path, help="Path to query image")
  p_sim.add_argument(
      "--include-self",
      action="store_true",
      help="Include the query image itself in results",
  )
  _add_search_output_flags(p_sim)
  _add_common_flags(p_sim)

  p_status = sub.add_parser(
      "status", help="Show index statistics and backend health"
  )
  p_status.add_argument(
      "--json", action="store_true", help="Output status as JSON"
  )
  _add_common_flags(p_status)

  return parser


def _is_option_token(token: str) -> bool:
  return token.startswith("-") and len(token) > 1


def _normalize_argv(raw_args: list[str]) -> list[str]:
  """Expand `--thumb=<mode>`, hoist subcommands, and allow intermixed flags."""
  if "--" in raw_args:
    sep = raw_args.index("--")
    head_raw, tail = raw_args[:sep], raw_args[sep + 1 :]
    has_sep = True
  else:
    head_raw, tail = raw_args, []
    has_sep = False

  expanded: list[str] = []
  for token in head_raw:
    if token.startswith("--thumb="):
      expanded.extend(["--thumb", "--thumb-backend", token.split("=", 1)[1]])
    elif token.startswith("-t="):
      expanded.extend(["-t", "--thumb-backend", token.split("=", 1)[1]])
    else:
      expanded.append(token)

  opts: list[str] = []
  positionals: list[str] = []
  i = 0
  while i < len(expanded):
    tok = expanded[i]
    if tok in VALUE_FLAGS:
      opts.append(tok)
      if i + 1 < len(expanded):
        opts.append(expanded[i + 1])
        i += 2
        continue
    elif any(tok.startswith(f + "=") for f in VALUE_FLAGS):
      opts.append(tok)
    elif tok.startswith(SHORT_VALUE_PREFIXES) and len(tok) > 2:
      opts.append(tok)
    elif _is_option_token(tok):
      opts.append(tok)
    else:
      positionals.append(tok)
    i += 1

  if any(h in opts for h in ("-h", "--help")) and not (
      positionals and positionals[0] in SUBCOMMANDS
  ):
    return ["--help"]

  if positionals and positionals[0] in SUBCOMMANDS:
    cmd = positionals[0]
    rest_pos = positionals[1:]
  elif positionals or has_sep:
    cmd = "search"
    rest_pos = positionals
  else:
    return expanded

  all_pos = [*rest_pos, *tail]
  needs_sep = has_sep or any(_is_option_token(p) for p in all_pos)
  if all_pos and needs_sep:
    return [cmd, *opts, "--", *all_pos]
  return [cmd, *opts, *all_pos]


def _validate_search_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
  if args.limit < 1:
    parser.error("--limit must be >= 1")
  if not math.isfinite(args.min_score) or not -1.0 <= args.min_score <= 1.0:
    parser.error("--min-score must be between -1.0 and 1.0")
  if not 1 <= args.thumb_rows <= MAX_THUMB_ROWS:
    parser.error(f"--thumb-rows must be between 1 and {MAX_THUMB_ROWS}")


def create_backend(args: argparse.Namespace) -> EmbeddingBackend:
  if args.endpoint:
    return EmbeddingClient(base_url=args.endpoint, model_id=args.model)
  mode = "vision" if args.command in ("index", "similar") else "text"
  return LocalEmbeddingClient(
      model_id=args.model,
      model_dir=args.model_dir,
      mode=mode,
      device=args.device,
  )


def main(argv: list[str] | None = None) -> int:
  raw_args = _normalize_argv(list(sys.argv[1:] if argv is None else argv))

  parser = build_parser()
  args = parser.parse_args(raw_args)

  if args.command == "index" and args.batch_size < 1:
    parser.error("--batch-size must be >= 1")
  if args.command in ("search", "similar"):
    _validate_search_args(parser, args)
  if args.command == "search" and not " ".join(args.query).strip():
    parser.error("search query must not be empty")

  try:
    with create_backend(args) as client:
      if args.command == "index":
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
          dur = format_duration(summary["elapsed_seconds"])
          print(
              f"Indexed: {summary['embedded']} new, "
              f"{summary['up_to_date']} unchanged, "
              f"{summary['reused_hash']} hash-reused, "
              f"{summary['failed']} failed in {dur} -> {args.db}"
          )
        return 0 if summary["failed"] == 0 else 2

      if args.command == "search":
        query_text = " ".join(args.query).strip()
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
          state_lbl = "exists" if info["db_exists"] else "missing"
          print(f"Database:       {info['db_path']} ({state_lbl})")
          if info["db_exists"]:
            size_mb = info["db_size_bytes"] / (1024 * 1024)
            print(f"Root directory: {info['root_dir']}")
            print(
                f"Indexed model:  {info['indexed_model']} (dim={info['dim']})"
            )
            print(
                f"Images:         {info['embedded']}/{info['total_rows']} "
                f"embedded ({info['unique_hashes']} unique hashes, "
                f"{info['errors']} errors)"
            )
            print(f"DB size:        {size_mb:.2f} MB")
          if "weights_dir" in info:
            dl_state = (
                "downloaded"
                if info["weights_downloaded"]
                else "not downloaded yet"
            )
            print(f"Weights dir:    {info['weights_dir']} ({dl_state})")
          if not info["endpoint_reachable"]:
            endpoint_state = "UNAVAILABLE"
          elif info["model_matches"] and info["served_model"]:
            endpoint_state = f"ready (model: {info['served_model']!r})"
          else:
            endpoint_state = (
                "reachable (matching embedding model not found; "
                f"available: {info['available_models']})"
            )
          print(f"Backend:        {info['endpoint']} — {endpoint_state}")
        return 0
  except BrokenPipeError:
    try:
      fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
      return 0
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
      os.dup2(devnull, fd)
    finally:
      os.close(devnull)
    return 0
  except (EmbedError, OSError, sqlite3.Error, ValueError) as exc:
    print(f"imgsearch: {exc}", file=sys.stderr)
    return 2

  return 0


if __name__ == "__main__":
  sys.exit(main())
