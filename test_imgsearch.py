"""Tests for imgsearch endpoint contract, incremental indexing, and CLI search."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

import imgsearch

PNG_HEADER = b"\x89PNG\r\n\x1a\n"
JPEG_HEADER = b"\xff\xd8\xff\xe0"


def make_mock_transport(
    served_model: str = "ggmlorg-embeddinggemma-2-BF16",
    llama_models_shape: bool = False,
    reject_marker: bytes | None = None,
):
  """Build an httpx.MockTransport that enforces the llama.cpp multimodal contract."""
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
        # Text query: deterministically map text to a 4-D vector
        assert inp.startswith(imgsearch.QUERY_PREFIX), "query must carry asymmetric prefix"
        q = inp[len(imgsearch.QUERY_PREFIX) :]
        vec = _fake_vec_for_tag(q)
        return httpx.Response(200, json={"data": [{"embedding": vec}]})

      assert isinstance(inp, list), "image input must be a list"
      out = []
      for item in inp:
        # Enforce nested content[].image_url contract (flat shape 400s on llama.cpp)
        if not isinstance(item, dict) or "content" not in item:
          return httpx.Response(400, text="flat shape rejected")
        part = item["content"][0]
        assert part["type"] == "image_url"
        url = part["image_url"]["url"]
        assert url.startswith("data:image/")
        b64 = url.split(",", 1)[1]
        import base64

        raw = base64.b64decode(b64)
        if reject_marker and reject_marker in raw:
          return httpx.Response(400, text="corrupt image rejected")
        tag = raw.decode("latin1")
        out.append({"embedding": _fake_vec_for_tag(tag)})
      return httpx.Response(200, json={"data": out})

    return httpx.Response(404)

  return httpx.MockTransport(handler), calls


def _fake_vec_for_tag(tag: str) -> list[float]:
  if "terminal" in tag:
    return [2.0, 0.0, 0.0, 0.0]  # deliberately unnormalized to test defensive L2 norm
  if "browser" in tag:
    return [0.0, 3.0, 0.0, 0.0]
  if "chart" in tag:
    return [0.0, 0.0, 4.0, 0.0]
  return [0.0, 0.0, 0.0, 1.0]


def test_detect_mime():
  assert imgsearch.detect_mime(PNG_HEADER + b"abc") == "image/png"
  assert imgsearch.detect_mime(JPEG_HEADER + b"abc") == "image/jpeg"
  assert imgsearch.detect_mime(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"
  assert imgsearch.detect_mime(b"unknown", Path("foo.jpg")) == "image/jpeg"


def test_model_probe_and_mismatch():
  transport, _ = make_mock_transport(served_model="qwen3-embedding-8b", llama_models_shape=True)
  with imgsearch.EmbeddingClient(
      "http://mock:8091", "embeddinggemma-2", transport=transport
  ) as client:
    with pytest.raises(imgsearch.ModelMismatch):
      client.verify()


def test_index_incremental_dedup_and_search(tmp_path: Path):
  root = tmp_path / "lib"
  (root / "sub1").mkdir(parents=True)
  (root / "sub2").mkdir(parents=True)

  img_term = root / "sub1" / "term.png"
  img_term_dup = root / "sub2" / "term_copy.png"
  img_browser = root / "sub1" / "browser.jpg"
  img_chart = root / "sub2" / "chart.png"

  img_term.write_bytes(PNG_HEADER + b"terminal screenshot")
  img_term_dup.write_bytes(PNG_HEADER + b"terminal screenshot")  # exact sha256 duplicate
  img_browser.write_bytes(JPEG_HEADER + b"browser window")
  img_chart.write_bytes(PNG_HEADER + b"chart plot")

  db_path = tmp_path / "index.db"
  transport, calls = make_mock_transport(llama_models_shape=True)

  with imgsearch.EmbeddingClient(
      "http://mock:8091", "embeddinggemma-2", transport=transport
  ) as client:
    summary1 = imgsearch.build_index(root, db_path, client, batch_size=2, quiet=True)
    # 4 files total, but 1 is an exact sha256 duplicate in a later batch -> only 3 embedded!
    assert summary1["total_files"] == 4
    assert summary1["embedded"] + summary1["reused_hash"] == 4
    assert summary1["reused_hash"] == 1
    assert summary1["failed"] == 0

    calls_after_first = len(calls)

    # Second run with no changes should make 0 HTTP embedding calls
    summary2 = imgsearch.build_index(root, db_path, client, batch_size=2, quiet=True)
    assert summary2["up_to_date"] == 4
    assert summary2["embedded"] == 0
    assert len(calls) == calls_after_first

    # Text search for "terminal"
    hits = imgsearch.search_text("terminal", db_path, client, limit=5)
    assert len(hits) == 4
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert hits[1]["score"] == pytest.approx(1.0, abs=1e-5)
    assert {hits[0]["rel_path"], hits[1]["rel_path"]} == {"sub1/term.png", "sub2/term_copy.png"}

    # Dedup collapses exact duplicates
    dedup_hits = imgsearch.search_text("terminal", db_path, client, limit=5, dedup=True)
    assert len(dedup_hits) == 3
    assert dedup_hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert dedup_hits[1]["score"] == pytest.approx(0.0, abs=1e-5)

    # Folder filter
    sub2_hits = imgsearch.search_text("chart", db_path, client, folder="sub2", min_score=0.5)
    assert [h["rel_path"] for h in sub2_hits] == ["sub2/chart.png"]

    # Similar image search on an indexed image should NOT hit the endpoint!
    calls_before_sim = len(calls)
    sim_hits = imgsearch.search_similar(img_term, db_path, client, limit=3)
    assert len(calls) == calls_before_sim
    assert sim_hits[0]["rel_path"] == "sub2/term_copy.png"
    assert sim_hits[0]["score"] == pytest.approx(1.0, abs=1e-5)

    # Deleting a file prunes it on re-index
    img_chart.unlink()
    summary3 = imgsearch.build_index(root, db_path, client, quiet=True)
    assert summary3["pruned"] == 1
    assert summary3["up_to_date"] == 3


def test_batch_rejection_falls_back_to_serial_and_permissions_recover(tmp_path: Path):
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
      summary = imgsearch.build_index(root, db_path, client, batch_size=8, quiet=True)
      assert summary["embedded"] == 1
      # 1 rejected by server + 1 unreadable (unless running as root)
      expected_failed = 1 if os.geteuid() == 0 else 2
      assert summary["failed"] == expected_failed

      if os.geteuid() != 0:
        # Fix permissions: next index pass should automatically pick up `locked`!
        locked.chmod(0o644)
        summary2 = imgsearch.build_index(root, db_path, client, batch_size=8, quiet=True)
        assert summary2["embedded"] == 1
        assert summary2["up_to_date"] == 1
        assert summary2["skipped_errors"] == 1  # `b_bad.png` stays skipped unless --retry-errors
  finally:
    locked.chmod(0o644)


def test_cli_shorthand_and_output_modes(tmp_path: Path, monkeypatch, capsys):
  root = tmp_path / "lib"
  root.mkdir()
  (root / "terminal_err.png").write_bytes(PNG_HEADER + b"terminal traceback")
  (root / "browser_tab.png").write_bytes(PNG_HEADER + b"browser page")
  db_path = tmp_path / "index.db"

  transport, _ = make_mock_transport()
  orig_client = imgsearch.EmbeddingClient

  def patched_client(base_url, model_id=imgsearch.DEFAULT_MODEL):
    return orig_client(base_url=base_url, model_id=model_id, transport=transport)

  monkeypatch.setattr(imgsearch, "EmbeddingClient", patched_client)

  rc = imgsearch.main(
      ["index", str(root), "--db", str(db_path), "--endpoint", "http://mock:8091", "-q"]
  )
  assert rc == 0

  # Shorthand `imgsearch "terminal" ...` dispatches to `search`
  rc = imgsearch.main(
      ["terminal", "--db", str(db_path), "--endpoint", "http://mock:8091", "--json", "-n", "1"]
  )
  assert rc == 0
  out = json.loads(capsys.readouterr().out)
  assert len(out) == 1
  assert out[0]["rel_path"] == "terminal_err.png"

  # `-l` / `--paths-only`
  rc = imgsearch.main(
      [
          "search",
          "browser",
          "--db",
          str(db_path),
          "--endpoint",
          "http://mock:8091",
          "-l",
          "--abs",
          "-n",
          "1",
      ]
  )
  assert rc == 0
  line = capsys.readouterr().out.strip()
  assert line == str((root / "browser_tab.png").resolve())

  # `status --json`
  rc = imgsearch.main(["status", "--db", str(db_path), "--endpoint", "http://mock:8091", "--json"])
  assert rc == 0
  st = json.loads(capsys.readouterr().out)
  assert st["embedded"] == 2
  assert st["endpoint_reachable"] is True


def test_detect_thumb_backend_and_osc8():
  assert imgsearch.detect_thumb_backend({"KITTY_WINDOW_ID": "1", "TERM": "xterm-kitty"}) == "kitty"
  assert (
      imgsearch.detect_thumb_backend({"KONSOLE_VERSION": "240800", "TERM": "xterm-256color"})
      == "kitty"
  )
  assert (
      imgsearch.detect_thumb_backend({"KITTY_WINDOW_ID": "1", "TMUX": "/tmp/tmux-1000/default"})
      == "ansi"
  )
  assert imgsearch.detect_thumb_backend({"IMGSEARCH_THUMB": "sixel", "TMUX": "1"}) == "sixel"
  assert imgsearch.detect_thumb_backend({"TERM": "dumb"}) == "ansi"

  assert imgsearch.supports_osc8({"TERM": "xterm-kitty"}, isatty=True) is True
  assert imgsearch.supports_osc8({"TERM": "dumb"}, isatty=True) is False
  assert imgsearch.supports_osc8({"TERM": "xterm-kitty"}, isatty=False) is False

  linked = imgsearch.osc8_link("/tmp/my file.png", "my file.png")
  assert linked == "\x1b]8;;file:///tmp/my%20file.png\x1b\\my file.png\x1b]8;;\x1b\\"


def test_thumbnail_renderers_and_cli_flags(tmp_path: Path, monkeypatch, capsys):
  from PIL import Image

  root = tmp_path / "lib"
  root.mkdir()
  sample = root / "terminal_sample.png"
  # Create a valid RGBA PNG so Pillow can load it AND our mock embedder sees "terminal" in raw bytes
  im = Image.new("RGBA", (64, 32), (20, 160, 220, 200))
  im.save(sample, format="PNG")

  kitty_out = imgsearch.render_thumbnail(sample, backend="kitty", rows=6, term_cols=80)
  assert kitty_out.startswith("    \x1b_Ga=T,f=100,r=6,q=2,m=")
  assert kitty_out.endswith("\x1b\\\n")

  ansi_out = imgsearch.render_thumbnail(sample, backend="ansi", rows=4, term_cols=80)
  ansi_lines = ansi_out.splitlines()
  assert len(ansi_lines) == 4
  assert "▀" in ansi_lines[0]
  assert ansi_lines[0].endswith("\x1b[0m")

  if imgsearch.shutil.which("img2sixel"):
    sixel_out = imgsearch.render_thumbnail(sample, backend="sixel", rows=4, term_cols=80)
    assert "\x1bP" in sixel_out

  # Verify CLI `-t kitty` treats `kitty` as query text, while `--thumb=kitty` sets the backend
  db_path = tmp_path / "index.db"
  transport, _ = make_mock_transport()
  orig_client = imgsearch.EmbeddingClient

  def patched_client(base_url, model_id=imgsearch.DEFAULT_MODEL):
    return orig_client(base_url=base_url, model_id=model_id, transport=transport)

  monkeypatch.setattr(imgsearch, "EmbeddingClient", patched_client)
  assert (
      imgsearch.main(
          ["index", str(root), "--db", str(db_path), "--endpoint", "http://mock:8091", "-q"]
      )
      == 0
  )

  assert (
      imgsearch.main(
          ["-t", "kitty", "--db", str(db_path), "--endpoint", "http://mock:8091", "-n", "1"]
      )
      == 0
  )
  out_auto = capsys.readouterr().out
  assert "terminal_sample.png" in out_auto
  assert ("▀" in out_auto) or ("\x1b_Ga=T" in out_auto) or ("\x1bP" in out_auto)

  assert (
      imgsearch.main(
          [
              "--thumb=kitty",
              "--thumb-rows",
              "5",
              "terminal",
              "--db",
              str(db_path),
              "--endpoint",
              "http://mock:8091",
          ]
      )
      == 0
  )
  out_kitty = capsys.readouterr().out
  assert "\x1b_Ga=T,f=100,r=5,q=2" in out_kitty


def test_readonly_db_and_directory_search(tmp_path: Path):
  import sqlite3

  ro_dir = tmp_path / "shared_dir"
  ro_dir.mkdir()
  img = ro_dir / "terminal_shot.png"
  img.write_bytes(PNG_HEADER + b"terminal view")
  db_path = ro_dir / "index.db"

  transport, _ = make_mock_transport()
  with imgsearch.EmbeddingClient(
      "http://mock:8091", "embeddinggemma-2", transport=transport
  ) as client:
    imgsearch.build_index(ro_dir, db_path, client, quiet=True)

    # Case 1: Normal (DELETE-mode) DB in a strictly read-only directory (0555) + 0444 DB
    db_path.chmod(0o444)
    ro_dir.chmod(0o555)
    try:
      hits = imgsearch.search_text("terminal", db_path, client, limit=5)
      assert len(hits) == 1
      assert hits[0]["rel_path"] == "terminal_shot.png"
      sim = imgsearch.search_similar(img, db_path, client, include_self=True)
      assert len(sim) == 1
      st = imgsearch.get_status(db_path, client)
      assert st["embedded"] == 1
    finally:
      ro_dir.chmod(0o755)
      db_path.chmod(0o644)

    # Case 2: Legacy WAL-flagged DB in a strictly read-only directory (fallback to immutable=1)
    raw_conn = sqlite3.connect(db_path)
    raw_conn.execute("PRAGMA journal_mode=WAL")
    raw_conn.close()
    db_path.chmod(0o444)
    ro_dir.chmod(0o555)
    try:
      hits_wal = imgsearch.search_text("terminal", db_path, client, limit=5)
      assert len(hits_wal) == 1
    finally:
      ro_dir.chmod(0o755)
      db_path.chmod(0o644)


def test_auto_select_first_embedding_model_and_cwd_defaults(tmp_path: Path, monkeypatch, capsys):
  calls: list[tuple[str, dict]] = []

  def router_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
      return httpx.Response(
          200,
          json={
              "models": [
                  {"name": "gemma-4", "capabilities": ["completion", "multimodal"]},
                  {"name": "embeddinggemma-2", "capabilities": ["completion", "multimodal"]},
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
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0, 0.0, 0.0]}]})
      return httpx.Response(
          200,
          json={"data": [{"embedding": [1.0, 0.0, 0.0, 0.0]} for _ in body["input"]]},
      )
    return httpx.Response(404)

  transport = httpx.MockTransport(router_handler)
  orig_client = imgsearch.EmbeddingClient

  def patched_client(base_url, model_id=imgsearch.DEFAULT_MODEL):
    return orig_client(base_url=base_url, model_id=model_id, transport=transport)

  monkeypatch.setattr(imgsearch, "EmbeddingClient", patched_client)
  monkeypatch.chdir(tmp_path)

  (tmp_path / "shot.png").write_bytes(PNG_HEADER + b"terminal")

  # Run `imgsearch index` with --endpoint, NO --model, NO --db, NO root!
  assert imgsearch.main(["index", "--endpoint", "http://localhost:8080", "-q"]) == 0
  assert (tmp_path / "index.db").exists()
  assert calls[0][0] == "http://localhost:8080/v1/embeddings"
  assert calls[0][1]["model"] == "embeddinggemma-2"  # skipped gemma-4, picked first embed model

  # Run `imgsearch "terminal" --endpoint ...` -> uses cwd `index.db` + indexed model
  assert imgsearch.main(["terminal", "--endpoint", "http://localhost:8080", "--json"]) == 0
  out = json.loads(capsys.readouterr().out)
  assert len(out) == 1
  assert out[0]["path"] == "shot.png"
  assert calls[-1][1]["model"] == "embeddinggemma-2"

  # Endpoint with ONLY chat models raises ModelMismatch when --model is omitted
  chat_only = httpx.MockTransport(
      lambda req: httpx.Response(200, json={"data": [{"id": "gemma-4"}, {"id": "llama-3"}]})
  )
  with orig_client("http://localhost:8080", None, transport=chat_only) as client:
    with pytest.raises(imgsearch.ModelMismatch, match="no embedding model found"):
      client.verify()


def test_local_default_weights_dir_and_cross_backend_interop(tmp_path: Path, monkeypatch, capsys):
  import huggingface_hub
  from PIL import Image
  import local_embed

  assert local_embed.build_selective_config_kwargs(
      "google/embeddinggemma-2", need_vision=False
  ) == {
      "vision_config": None,
      "audio_config": None,
  }
  assert local_embed.build_selective_config_kwargs("google/embeddinggemma-2", need_vision=True) == {
      "audio_config": None,
  }
  assert local_embed.build_selective_config_kwargs("BAAI/bge-m3", need_vision=False) == {}

  # Verify `resolve_weights_dir` defaults to `./models/<slug>` and respects `--model-dir`
  monkeypatch.chdir(tmp_path)
  assert local_embed.resolve_weights_dir("google/embeddinggemma-2") == Path(
      "models/embeddinggemma-2"
  )
  assert local_embed.resolve_weights_dir("google/embeddinggemma-2", tmp_path / "custom_w") == (
      tmp_path / "custom_w"
  )

  # Verify `ensure_weights_downloaded` downloads once and skips once files exist
  dl_calls: list[tuple[str, str]] = []

  def fake_snapshot_download(repo_id: str, local_dir: str, **_kwargs):
    dl_calls.append((repo_id, local_dir))
    p = Path(local_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.json").write_text('{"model_type": "embedding_gemma2"}', encoding="utf-8")
    (p / "model.safetensors").write_bytes(b"weights")
    return local_dir

  monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
  target_wdir = tmp_path / "models" / "embeddinggemma-2"
  local_embed.ensure_weights_downloaded("google/embeddinggemma-2", target_wdir)
  local_embed.ensure_weights_downloaded("google/embeddinggemma-2", target_wdir)
  assert len(dl_calls) == 1  # second call skipped because weights already exist!
  # Even with an unrelated model name, config.json detection triggers Gemma 2 selective loading
  assert local_embed.build_selective_config_kwargs(
      "custom-name", need_vision=False, weights_dir=target_wdir
  ) == {"vision_config": None, "audio_config": None}

  # Put a dummy PNG inside `models/` to verify `discover_images` ignores model weight directories
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
        tag = "terminal" if r > 100 else "browser"
        vecs.append(_fake_vec_for_tag(tag))
      return np.asarray(vecs, dtype=np.float32)

  def fake_loader(hf_model_id: str, weights_dir: Path, need_vision: bool, device: str | None):
    load_calls.append((hf_model_id, weights_dir, need_vision, device))
    return FakeSentenceTransformer()

  img1 = tmp_path / "first.png"
  Image.new("RGB", (16, 16), (200, 10, 10)).save(img1, format="PNG")
  db_path = tmp_path / "index.db"

  # 1. Index img1 via remote llama.cpp client (stamps 'ggmlorg-embeddinggemma-2-BF16')
  transport, _ = make_mock_transport(served_model="ggmlorg-embeddinggemma-2-BF16")
  with imgsearch.EmbeddingClient("http://mock:8091", "embeddinggemma-2", transport=transport) as rc:
    s1 = imgsearch.build_index(tmp_path, db_path, rc, quiet=True)
    assert s1["embedded"] == 1  # `models/.../banner.png` was ignored!

  # 2. Search locally (text-only: need_vision=False!) against the remote-indexed DB
  with local_embed.LocalEmbeddingClient(mode="text", device="cpu", loader=fake_loader) as lc:
    hits = imgsearch.search_text("terminal", db_path, lc, limit=5)
    assert len(hits) == 1
    assert hits[0]["rel_path"] == "first.png"
    assert load_calls == [
        ("google/embeddinggemma-2", Path("models/embeddinggemma-2"), False, "cpu")
    ]

  # 3. `similar` on an already-indexed image loads ZERO weights!
  load_calls.clear()
  with local_embed.LocalEmbeddingClient(mode="vision", device="cpu", loader=fake_loader) as lc_sim:
    sim = imgsearch.search_similar(img1, db_path, lc_sim, include_self=True)
    assert len(sim) == 1
    assert load_calls == []

  # 4. Add img2 and run local `build_index` -> keeps img1 up_to_date and embeds only img2!
  img2 = tmp_path / "second.png"
  Image.new("RGB", (16, 16), (10, 200, 10)).save(img2, format="PNG")
  with local_embed.LocalEmbeddingClient(mode="vision", device="cpu", loader=fake_loader) as lc_idx:
    s2 = imgsearch.build_index(tmp_path, db_path, lc_idx, quiet=True)
    assert s2["up_to_date"] == 1
    assert s2["embedded"] == 1
    assert load_calls == [("google/embeddinggemma-2", Path("models/embeddinggemma-2"), True, "cpu")]

  # 5. CLI with NO `--endpoint` defaults to local backend and honors `--model-dir`!
  load_calls.clear()
  orig_local = imgsearch.LocalEmbeddingClient
  monkeypatch.setattr(
      imgsearch,
      "LocalEmbeddingClient",
      lambda model_id=None, model_dir=None, mode="text", device=None: orig_local(
          model_id=model_id, model_dir=model_dir, mode=mode, device=device, loader=fake_loader
      ),
  )
  custom_dir = tmp_path / "my_weights"
  assert (
      imgsearch.main(["terminal", "--model-dir", str(custom_dir), "--device", "cpu", "--json"]) == 0
  )
  cli_hits = json.loads(capsys.readouterr().out)
  assert len(cli_hits) == 2
  assert load_calls == [("google/embeddinggemma-2", custom_dir, False, "cpu")]
