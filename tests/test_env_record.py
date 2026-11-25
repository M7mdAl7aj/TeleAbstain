from __future__ import annotations

import json

from teleabstain.env_record import TRACKED, EnvRecord, capture, mxfp4_state, write_sidecar


def test_capture_records_every_tracked_package_key() -> None:
    """A missing package is recorded as None, never omitted - absence is provenance too."""
    rec = capture()
    assert set(rec.packages) == set(TRACKED)


def test_hash_is_stable_for_the_same_environment() -> None:
    assert capture().env_hash == capture().env_hash


def test_hash_moves_when_a_tracked_version_moves() -> None:
    a = capture()
    b = capture()
    b.packages = dict(b.packages, torch="0.0.0-not-a-real-version")
    assert b.compute_hash() != a.env_hash


def test_note_text_is_excluded_from_the_hash() -> None:
    """Prose must not move the hash; only the state it describes may."""
    a = capture()
    b = capture()
    b.mxfp4 = dict(b.mxfp4, note="reworded entirely")
    assert b.compute_hash() == a.env_hash


def test_mxfp4_path_follows_the_kernels_package() -> None:
    st = mxfp4_state()
    expected = "native-mxfp4" if st["kernels_installed"] else "dequantized-to-bf16"
    assert st["mxfp4_path"] == expected


def test_sidecar_round_trips(tmp_path) -> None:
    p = tmp_path / "env.json"
    rec = write_sidecar(p)
    loaded = json.loads(p.read_text())
    assert loaded["env_hash"] == rec.env_hash
    assert EnvRecord(**{k: v for k, v in loaded.items()}).env_hash == rec.env_hash
