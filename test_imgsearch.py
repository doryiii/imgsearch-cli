"""Tests for imgsearch contract, backends, storage, CLI, and security."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sqlite3

import httpx
import numpy as np
import pytest

import embed_base
import imgsearch
import local_embed

PNG_HEADER = b"\x89PNG\r\n\x1a\n"
JPEG_HEADER = b"\xff\xd8\xff\xe0"


def make_mock_transport(
    served_model: str = "ggmlorg-embeddinggemma-2-BF16",
    llama_models_shape: bool = False,
    reject_marker: bytes | None = None,
):
  """Build an httpx.MockTransport enforcing the llama.cpp contract."""
  calls: list[dict] = []

  def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
      if llama_models_shape:
        body = {"models": [{"name": served_model, "model": served_model}]}
      else:
        body = {"data": [{"id": served_model}]}
      return httpx.Response(200, json=body)

    if request.url.path == "/v1/embeddings":
      payload = json.loads(request.content.decode("utf-8"))
      calls.append(payload)
      inp = payload["input"]

      if isinstance(inp, str):
        if "embeddinggemma" in served_model.lower():
          assert inp.startswith(imgsearch.QUERY_PREFIX)
          q = inp[len(imgsearch.QUERY_PREFIX) :]
        else:
          assert not inp.startswith(imgsearch.QUERY_PREFIX)
          q = inp
        vec = _fake_vec_for_tag(q)
        return httpx.Response(200, json={"data": [{"embedding": vec}]})

      assert isinstance(inp, list)
      out = []
      for item in inp:
        if not isinstance(item, dict) or "content" not in item:
          return httpx.Response(400, text="flat shape rejected")
        part = item["content"][0]
        assert part["type"] == "image_url"
        url = part["image_url"]["url"]
        assert url.startswith("data:image/")
        raw = base64.b64decode(url.split(",", 1)[1])
        if reject_marker and reject_marker in raw:
          return httpx.Response(400, text="corrupt image rejected")
        out.append({"embedding": _fake_vec_for_tag(raw.decode("latin1"))})
      return httpx.Response(200, json={"data": out})

    return httpx.Response(404)

  return httpx.MockTransport(handler), calls


def _fake_vec_for_tag(tag: str) -> list[float]:
  if "terminal" in tag:
    return [2.0, 0.0, 0.0, 0.0]
  if "browser" in tag:
    return [0.0, 3.0, 0.0, 0.0]
  if "chart" in tag:
    return [0.0, 0.0, 4.0, 0.0]
  return [0.0, 0.0, 0.0, 1.0]


def test_detect_mime_and_normalize():
  assert imgsearch.detect_mime(PNG_HEADER + b"abc") == "image/png"
  assert imgsearch.detect_mime(JPEG_HEADER + b"abc") == "image/jpeg"
  assert imgsearch.detect_mime(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"
  assert imgsearch.detect_mime(b"unknown", Path("foo.jpg")) == "image/jpeg"

  with pytest.raises(imgsearch.EmbedError):
    imgsearch.normalize([float("nan"), 1.0])
  with pytest.raises(imgsearch.EmbedError):
    imgsearch.normalize([float("inf"), 1.0])


def test_model_compatibility_and_gguf_resolution():
  assert embed_base.models_compatible(
      "embeddinggemma-2", "google/embeddinggemma-2"
  )
  assert embed_base.models_compatible(
      "ggmlorg-embeddinggemma-2-BF16", "embeddinggemma-2-Q8_0"
  )
  # Must NOT conflate v1 and v2 or broad substrings
  assert not embed_base.models_compatible("embeddinggemma", "embeddinggemma-2")
  assert not embed_base.models_compatible("embed", "embeddinggemma-2")
  assert not embed_base.models_compatible(".", "model.gguf")
  assert embed_base.models_compatible("BAAI/bge-m3", "bge-m3")

  assert (
      embed_base.resolve_local_hf_model("/models/embeddinggemma-2-Q8_0.gguf")
      == "google/embeddinggemma-2"
  )
  assert (
      embed_base.resolve_local_hf_model("ggml-org/embeddinggemma-2-GGUF")
      == "google/embeddinggemma-2"
  )


def test_model_probe_mismatch_and_remote_parity(tmp_path: Path):
  t_qwen, _ = make_mock_transport(
      served_model="qwen3-embedding-8b", llama_models_shape=True
  )
  with imgsearch.EmbeddingClient(
      "http://mock:8091", "embeddinggemma-2", transport=t_qwen
  ) as client:
    info = client.probe()
    assert info["matches"] is False
    assert info["served_model"] is None
    with pytest.raises(imgsearch.ModelMismatch):
      client.verify()

  # Index with embeddinggemma-2, then query remotely with incompatible --model
  root = tmp_path / "lib"
  root.mkdir()
  (root / "a.png").write_bytes(PNG_HEADER + b"terminal")
  db_path = tmp_path / "index.db"
  t_gemma, _ = make_mock_transport(served_model="embeddinggemma-2")
  with imgsearch.EmbeddingClient(
      "http://mock:8091", None, transport=t_gemma
  ) as c1:
    imgsearch.build_index(root, db_path, c1, quiet=True)

  with imgsearch.EmbeddingClient(
      "http://mock:8091", "qwen3-embedding-8b", transport=t_qwen
  ) as c2:
    with pytest.raises(imgsearch.ModelMismatch):
      imgsearch.search_text("terminal", db_path, c2)


def test_index_incremental_dedup_folder_wildcards_and_similar(tmp_path: Path):
  root = tmp_path / "lib"
  (root / "2026_04").mkdir(parents=True)
  (root / "2026-04").mkdir(parents=True)
  # User photo folder named `models/` must NOT be skipped!
  (root / "models").mkdir(parents=True)

  img_term = root / "2026_04" / "term.png"
  img_term_dup = root / "models" / "term_copy.png"
  img_browser = root / "2026-04" / "browser.jpg"
  img_chart = root / "2026_04" / "chart.png"

  img_term.write_bytes(PNG_HEADER + b"terminal screenshot")
  img_term_dup.write_bytes(PNG_HEADER + b"terminal screenshot")
  img_browser.write_bytes(JPEG_HEADER + b"browser window")
  img_chart.write_bytes(PNG_HEADER + b"chart plot")

  ext_img = tmp_path / "external_real.png"
  ext_img.write_bytes(PNG_HEADER + b"chart plot external")
  sym_img = root / "2026_04" / "sym_chart.png"
  sym_img.symlink_to(ext_img)

  db_path = tmp_path / "index.db"
  transport, calls = make_mock_transport(llama_models_shape=True)

  with imgsearch.EmbeddingClient(
      "http://mock:8091", "embeddinggemma-2", transport=transport
  ) as client:
    s1 = imgsearch.build_index(root, db_path, client, batch_size=2, quiet=True)
    assert s1["total_files"] == 5
    assert s1["reused_hash"] == 1
    assert s1["failed"] == 0

    calls_after_first = len(calls)
    s2 = imgsearch.build_index(root, db_path, client, batch_size=2, quiet=True)
    assert s2["up_to_date"] == 5
    assert len(calls) == calls_after_first

    # `--folder 2026_04` must NOT match `2026-04` (`_` escaped in LIKE)
    f_hits = imgsearch.search_text("browser", db_path, client, folder="2026_04")
    assert all("2026-04" not in h["rel_path"] for h in f_hits)

    # `similar` on symlink inside root hits DB cache & excludes self
    calls_before_sim = len(calls)
    sim_hits = imgsearch.search_similar(sym_img, db_path, client, limit=5)
    assert len(calls) == calls_before_sim
    assert all(h["rel_path"] != "2026_04/sym_chart.png" for h in sim_hits)

    # Modifying `img_term` on disk invalidates `search_similar`'s DB cache
    img_term.write_bytes(PNG_HEADER + b"browser updated bytes")
    sim_updated = imgsearch.search_similar(img_term, db_path, client, limit=1)
    assert len(calls) == calls_before_sim + 1
    assert sim_updated[0]["rel_path"] == "2026-04/browser.jpg"


def test_batch_rejection_systemic_errors_and_out_of_order_index(tmp_path: Path):
  root = tmp_path / "lib"
  root.mkdir()
  good = root / "a_good.png"
  bad = root / "b_bad.png"
  locked = root / "c_locked.png"

  good.write_bytes(PNG_HEADER + b"browser good")
  bad.write_bytes(PNG_HEADER + b"BAD_PAYLOAD")
  locked.write_bytes(PNG_HEADER + b"terminal locked")
  locked.chmod(0o000)

  db_path = tmp_path / "index.db"
  transport, _ = make_mock_transport(reject_marker=b"BAD_PAYLOAD")

  try:
    with imgsearch.EmbeddingClient(
        "http://mock:8091", "embeddinggemma-2", transport=transport
    ) as client:
      s1 = imgsearch.build_index(root, db_path, client, quiet=True)
      assert s1["embedded"] == 1
      expected_failed = 1 if os.geteuid() == 0 else 2
      assert s1["failed"] == expected_failed

      if os.geteuid() != 0:
        locked.chmod(0o644)
        s2 = imgsearch.build_index(root, db_path, client, quiet=True)
        assert s2["embedded"] == 1
        assert s2["up_to_date"] == 1
        assert s2["skipped_errors"] == 1
  finally:
    locked.chmod(0o644)

  # Systemic 400 (missing mmproj) must raise EndpointUnreachable, NOT poison DB
  db_sys = tmp_path / "sys.db"
  no_mmproj = httpx.MockTransport(
      lambda req: (
          httpx.Response(200, json={"data": [{"id": "embeddinggemma-2"}]})
          if req.url.path == "/v1/models"
          else httpx.Response(400, text="image input is not supported (mmproj)")
      )
  )
  with imgsearch.EmbeddingClient(
      "http://mock:8091", None, transport=no_mmproj
  ) as bad_srv:
    with pytest.raises(imgsearch.EndpointUnreachable):
      imgsearch.build_index(root, db_sys, bad_srv, quiet=True)
  conn = sqlite3.connect(db_sys)
  assert conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 0
  conn.close()

  # Out-of-order `data[].index` must be sorted back to input order
  ooo_transport = httpx.MockTransport(
      lambda req: (
          httpx.Response(200, json={"data": [{"id": "embeddinggemma-2"}]})
          if req.url.path == "/v1/models"
          else httpx.Response(
              200,
              json={
                  "data": [
                      {"index": 1, "embedding": [0.0, 1.0]},
                      {"index": 0, "embedding": [1.0, 0.0]},
                  ]
              },
          )
      )
  )
  with imgsearch.EmbeddingClient(
      "http://mock:8091", None, transport=ooo_transport
  ) as ooo_client:
    vecs = ooo_client.embed_images([(b"a", "image/png"), (b"b", "image/png")])
    assert list(vecs[0]) == pytest.approx([1.0, 0.0])
    assert list(vecs[1]) == pytest.approx([0.0, 1.0])


def test_cli_argv_normalization_validation_and_modes(
    tmp_path: Path, monkeypatch, capsys
):
  root = tmp_path / "lib"
  root.mkdir()
  img_a = root / "terminal_err.png"
  img_b = root / "browser_tab.png"
  img_a.write_bytes(PNG_HEADER + b"terminal traceback")
  img_b.write_bytes(PNG_HEADER + b"browser page")
  db_path = tmp_path / "index.db"

  transport, _ = make_mock_transport()
  orig_client = imgsearch.EmbeddingClient
  monkeypatch.setattr(
      imgsearch,
      "EmbeddingClient",
      lambda base_url, model_id=None: orig_client(
          base_url=base_url, model_id=model_id, transport=transport
      ),
  )

  # Global flags BEFORE `index` and `status` must NOT misdispatch to `search`!
  rc = imgsearch.main(
      [
          "--db",
          str(db_path),
          "--endpoint",
          "http://mock:8091",
          "index",
          str(root),
          "-q",
      ]
  )
  assert rc == 0

  rc = imgsearch.main(
      [
          "--db",
          str(db_path),
          "--endpoint",
          "http://mock:8091",
          "status",
          "--json",
      ]
  )
  assert rc == 0
  st = json.loads(capsys.readouterr().out)
  assert st["embedded"] == 2
  assert st["model_matches"] is True

  # Interleaved flags (`terminal -n 1 traceback`) work cleanly
  rc = imgsearch.main(
      [
          "terminal",
          "-n",
          "1",
          "traceback",
          "--db",
          str(db_path),
          "--endpoint",
          "http://mock:8091",
          "--json",
      ]
  )
  assert rc == 0
  out = json.loads(capsys.readouterr().out)
  assert out[0]["rel_path"] == "terminal_err.png"

  # `--` stops `--thumb=` expansion
  norm = imgsearch._normalize_argv(["search", "--", "--thumb=literal"])
  assert norm == ["search", "--", "--thumb=literal"]

  # `-0` / `--null` mode and `similar` CLI subcommand
  rc = imgsearch.main(
      [
          "similar",
          str(img_a),
          "--include-self",
          "--db",
          str(db_path),
          "--endpoint",
          "http://mock:8091",
          "-0",
          "-n",
          "1",
      ]
  )
  assert rc == 0
  nul_out = capsys.readouterr().out
  assert nul_out.endswith("\0")

  # Invalid numeric flags exit with code 2
  for bad_argv in (
      ["cat", "-n", "0", "--db", str(db_path)],
      ["cat", "--min-score", "nan", "--db", str(db_path)],
      ["cat", "--min-score", "2.0", "--db", str(db_path)],
      ["cat", "--thumb-rows", "0", "--db", str(db_path)],
  ):
    with pytest.raises(SystemExit) as exc_info:
      imgsearch.main(bad_argv)
    assert exc_info.value.code == 2


def test_tty_escape_sanitization_and_osc8():
  assert (
      imgsearch.detect_thumb_backend(
          {"KITTY_WINDOW_ID": "1", "TERM": "xterm-kitty"}
      )
      == "kitty"
  )
  assert (
      imgsearch.detect_thumb_backend(
          {"KONSOLE_VERSION": "240800", "TERM": "xterm-256color"}
      )
      == "kitty"
  )
  assert (
      imgsearch.detect_thumb_backend(
          {"KITTY_WINDOW_ID": "1", "TMUX": "/tmp/tmux-1000/default"}
      )
      == "ansi"
  )
  assert (
      imgsearch.detect_thumb_backend({"IMGSEARCH_THUMB": "sixel", "TMUX": "1"})
      == "sixel"
  )

  # Malicious control characters in label cannot escape OSC 8 or spoof lines
  evil_name = "bad\x1b\\inject\x07\r\nfile.png"
  linked = imgsearch.osc8_link("/tmp/safe.png", evil_name)
  assert linked.count("\x1b\\") == 2
  assert "\x07" not in linked
  assert "\n" not in linked
  assert "\ufffd" in linked


def test_thumbnail_renderers_and_corrupt_graceful_fallback(
    tmp_path: Path, monkeypatch, capsys
):
  from PIL import Image

  root = tmp_path / "lib"
  root.mkdir()
  sample = root / "terminal_sample.png"
  # Palette image with byte transparency verifies clean RGBA compositing
  pal = Image.new("P", (16, 16), 0)
  pal.putpalette([20, 160, 220] + [0, 0, 0] * 255)
  pal.info["transparency"] = b"\x00"
  pal.save(sample, format="PNG")

  kitty_out = imgsearch.render_thumbnail(
      sample, backend="kitty", rows=6, term_cols=80
  )
  assert kitty_out.startswith("    \x1b_Ga=T,f=100,r=6,q=2,m=")
  assert kitty_out.endswith("\x1b\\\n")

  ansi_out = imgsearch.render_thumbnail(
      sample, backend="ansi", rows=4, term_cols=80
  )
  ansi_lines = ansi_out.splitlines()
  assert len(ansi_lines) == 4
  assert "▀" in ansi_lines[0]

  if imgsearch.shutil.which("img2sixel") or imgsearch.shutil.which("magick"):
    sixel_out = imgsearch.render_thumbnail(
        sample, backend="sixel", rows=4, term_cols=80
    )
    assert "\x1bP" in sixel_out

  db_path = tmp_path / "index.db"
  transport, _ = make_mock_transport()
  orig_client = imgsearch.EmbeddingClient
  monkeypatch.setattr(
      imgsearch,
      "EmbeddingClient",
      lambda base_url, model_id=None: orig_client(
          base_url=base_url, model_id=model_id, transport=transport
      ),
  )
  assert (
      imgsearch.main(
          [
              "index",
              str(root),
              "--db",
              str(db_path),
              "--endpoint",
              "http://mock:8091",
              "-q",
          ]
      )
      == 0
  )

  # Corrupt the image AFTER indexing: `-t` prints warning instead of crashing!
  sample.write_bytes(b"not a valid png anymore")
  assert (
      imgsearch.main(
          [
              "--thumb=ansi",
              "terminal",
              "--db",
              str(db_path),
              "--endpoint",
              "http://mock:8091",
          ]
      )
      == 0
  )
  captured = capsys.readouterr()
  assert "terminal_sample.png" in captured.out
  assert "thumbnail unavailable" in captured.err


def test_readonly_special_chars_and_lock_safety(tmp_path: Path, monkeypatch):
  ro_dir = tmp_path / "dir with ?#%& spaces"
  ro_dir.mkdir()
  img = ro_dir / "terminal_shot.png"
  img.write_bytes(PNG_HEADER + b"terminal view")
  db_path = ro_dir / "index.db"

  transport, _ = make_mock_transport()
  with imgsearch.EmbeddingClient(
      "http://mock:8091", "embeddinggemma-2", transport=transport
  ) as client:
    imgsearch.build_index(ro_dir, db_path, client, quiet=True)

    # Lock contention (`database is locked`) must NOT fall back to immutable=1!
    monkeypatch.setattr(imgsearch, "BUSY_TIMEOUT_MS", 50)
    locker = sqlite3.connect(db_path)
    locker.isolation_level = None
    locker.execute("BEGIN EXCLUSIVE")
    try:
      with pytest.raises(sqlite3.OperationalError, match="locked"):
        imgsearch.open_db(db_path, readonly=True)
    finally:
      locker.execute("ROLLBACK")
      locker.close()
    monkeypatch.setattr(imgsearch, "BUSY_TIMEOUT_MS", 30_000)

    # Read-only directory (0555) + 0444 DB works for both DELETE and WAL DBs
    raw_conn = sqlite3.connect(db_path)
    raw_conn.execute("PRAGMA journal_mode=WAL")
    raw_conn.close()
    db_path.chmod(0o444)
    ro_dir.chmod(0o555)
    try:
      hits = imgsearch.search_text("terminal", db_path, client, limit=5)
      assert len(hits) == 1
    finally:
      ro_dir.chmod(0o755)
      db_path.chmod(0o644)


def test_auto_select_first_embedding_model_and_cwd_defaults(
    tmp_path: Path, monkeypatch, capsys
):
  calls: list[tuple[str, dict]] = []

  def router_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
      return httpx.Response(
          200,
          json={
              "models": [
                  {
                      "name": "gemma-4",
                      "capabilities": {"completion": True, "embeddings": False},
                  },
                  {
                      "name": "embeddinggemma-2",
                      "capabilities": ["completion", "multimodal"],
                  },
                  {"name": "qwenembed", "capabilities": ["completion"]},
              ],
              "data": [
                  {"id": "gemma-4"},
                  {"id": "embeddinggemma-2"},
                  {"id": "qwenembed"},
              ],
          },
      )
    if request.url.path == "/v1/embeddings":
      body = json.loads(request.content.decode("utf-8"))
      calls.append((str(request.url), body))
      if isinstance(body["input"], str):
        return httpx.Response(
            200, json={"data": [{"embedding": [1.0, 0.0, 0.0, 0.0]}]}
        )
      return httpx.Response(
          200,
          json={
              "data": [
                  {"embedding": [1.0, 0.0, 0.0, 0.0]} for _ in body["input"]
              ]
          },
      )
    return httpx.Response(404)

  transport = httpx.MockTransport(router_handler)
  orig_client = imgsearch.EmbeddingClient
  monkeypatch.setattr(
      imgsearch,
      "EmbeddingClient",
      lambda base_url, model_id=None: orig_client(
          base_url=base_url, model_id=model_id, transport=transport
      ),
  )
  monkeypatch.chdir(tmp_path)
  (tmp_path / "shot.png").write_bytes(PNG_HEADER + b"terminal")

  assert (
      imgsearch.main(["index", "--endpoint", "http://localhost:8080", "-q"])
      == 0
  )
  assert (tmp_path / "index.db").exists()
  assert calls[0][1]["model"] == "embeddinggemma-2"

  assert (
      imgsearch.main(
          ["terminal", "--endpoint", "http://localhost:8080", "--json"]
      )
      == 0
  )
  out = json.loads(capsys.readouterr().out)
  assert out[0]["path"] == "shot.png"


def test_local_weights_shards_and_cross_backend_interop(
    tmp_path: Path, monkeypatch, capsys
):
  import huggingface_hub
  from PIL import Image

  assert local_embed.build_selective_config_kwargs(
      "google/embeddinggemma-2", need_vision=False
  ) == {"vision_config": None, "audio_config": None}
  assert local_embed.build_selective_config_kwargs(
      "google/embeddinggemma-2", need_vision=True
  ) == {"audio_config": None}

  # Incomplete sharded index must return False until every shard file exists
  sharded_dir = tmp_path / "sharded"
  sharded_dir.mkdir()
  (sharded_dir / "config.json").write_text("{}", encoding="utf-8")
  (sharded_dir / "model.safetensors.index.json").write_text(
      json.dumps(
          {"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}
      ),
      encoding="utf-8",
  )
  (sharded_dir / "s1.safetensors").write_bytes(b"part1")
  assert local_embed.has_local_weights(sharded_dir) is False
  (sharded_dir / "s2.safetensors").write_bytes(b"part2")
  assert local_embed.has_local_weights(sharded_dir) is True

  monkeypatch.chdir(tmp_path)
  dl_calls: list[tuple[str, str]] = []

  def fake_snapshot_download(repo_id: str, local_dir: str, **_kwargs):
    dl_calls.append((repo_id, local_dir))
    p = Path(local_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.json").write_text(
        '{"model_type": "embedding_gemma2"}', encoding="utf-8"
    )
    (p / "model.safetensors").write_bytes(b"weights")
    return local_dir

  monkeypatch.setattr(
      huggingface_hub, "snapshot_download", fake_snapshot_download
  )
  target_wdir = tmp_path / "models" / "embeddinggemma-2"
  local_embed.ensure_weights_downloaded("google/embeddinggemma-2", target_wdir)
  local_embed.ensure_weights_downloaded("google/embeddinggemma-2", target_wdir)
  assert len(dl_calls) == 1
  (target_wdir / "banner.png").write_bytes(PNG_HEADER + b"ignore me")

  load_calls: list[tuple[str, Path, bool, str | None]] = []

  class FakeSentenceTransformer:
    prompts = {"SearchQuery": imgsearch.QUERY_PREFIX}

    def encode(
        self,
        inputs,
        prompt_name=None,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
      if isinstance(inputs, str):
        assert prompt_name == "SearchQuery"
        return np.asarray(_fake_vec_for_tag(inputs), dtype=np.float32)
      vecs = []
      for item in inputs:
        r, _g, _b = item["image"].getpixel((0, 0))
        vecs.append(_fake_vec_for_tag("terminal" if r > 100 else "browser"))
      return np.asarray(vecs, dtype=np.float32)

  def fake_loader(
      hf_model_id: str, weights_dir: Path, need_vision: bool, device: str | None
  ):
    load_calls.append((hf_model_id, weights_dir, need_vision, device))
    return FakeSentenceTransformer()

  img1 = tmp_path / "first.png"
  Image.new("RGB", (16, 16), (200, 10, 10)).save(img1, format="PNG")
  db_path = tmp_path / "index.db"

  # Remote GGUF path with `/` interoperates seamlessly with local search!
  transport, _ = make_mock_transport(
      served_model="/models/embeddinggemma-2-Q8_0.gguf"
  )
  with imgsearch.EmbeddingClient(
      "http://mock:8091", None, transport=transport
  ) as rc:
    s1 = imgsearch.build_index(tmp_path, db_path, rc, quiet=True)
    assert s1["embedded"] == 1

  with local_embed.LocalEmbeddingClient(
      mode="text", device="cpu", loader=fake_loader
  ) as lc:
    hits = imgsearch.search_text("terminal", db_path, lc, limit=5)
    assert len(hits) == 1
    assert load_calls == [
        (
            "google/embeddinggemma-2",
            Path("models/embeddinggemma-2"),
            False,
            "cpu",
        )
    ]

  orig_local = imgsearch.LocalEmbeddingClient

  def patched_local(model_id=None, model_dir=None, mode="text", device=None):
    return orig_local(
        model_id=model_id,
        model_dir=model_dir,
        mode=mode,
        device=device,
        loader=fake_loader,
    )

  monkeypatch.setattr(imgsearch, "LocalEmbeddingClient", patched_local)
  custom_dir = tmp_path / "my_weights"
  assert (
      imgsearch.main(
          [
              "terminal",
              "--model-dir",
              str(custom_dir),
              "--device",
              "cpu",
              "--json",
          ]
      )
      == 0
  )
  cli_hits = json.loads(capsys.readouterr().out)
  assert len(cli_hits) == 1
