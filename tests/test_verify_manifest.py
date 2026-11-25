import json
from pathlib import Path

from scripts.verify_manifest import main, verify

REPO = Path(__file__).resolve().parent.parent


def test_the_item_files_match_the_manifest() -> None:
    assert verify(REPO / "data", REPO / "manifest.json") == []
    assert main(["--items", str(REPO / "data"), "--manifest", str(REPO / "manifest.json")]) == 0


def test_a_changed_prompt_is_reported(tmp_path) -> None:
    for f in (REPO / "data").glob("*.jsonl"):
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
        if f.stem == "control":
            rows[0]["prompt"] += " "
        (tmp_path / f.name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    problems = verify(tmp_path, REPO / "manifest.json")
    assert any("does not hash" in p for p in problems)
    assert any("checksum" in p for p in problems)
    assert main(["--items", str(tmp_path), "--manifest", str(REPO / "manifest.json")]) == 1
