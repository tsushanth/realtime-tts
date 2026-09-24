import csv
import json

import pytest

from real_calls.review import apply_review, export_review


def _manifest(tmp_path):
    entries = [
        {"id": "c1_000", "call_id": "c1", "start_s": 0.0, "end_s": 3.0, "ref": "draft one", "verified": False, "keyterms": []},
        {"id": "c2_000", "call_id": "c2", "start_s": 1.0, "end_s": 4.0, "ref": "draft two", "verified": False, "keyterms": []},
    ]
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(entries))
    return p


def test_export_filters_by_call_id(tmp_path):
    m, t = _manifest(tmp_path), tmp_path / "review.tsv"
    export_review(str(m), str(t), call_ids={"c1"})
    rows = list(csv.DictReader(open(t), delimiter="\t"))
    assert [r["id"] for r in rows] == ["c1_000"]
    assert rows[0]["draft"] == "draft one" and rows[0]["verified_text"] == ""


def test_apply_review_marks_verified_and_parses_keyterms(tmp_path):
    m, t = _manifest(tmp_path), tmp_path / "review.tsv"
    export_review(str(m), str(t))
    rows = list(csv.DictReader(open(t), delimiter="\t"))
    rows[0]["verified_text"] = "hello this is Sushant"
    rows[0]["keyterms"] = "Sushant; 555 1234"
    with open(t, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    assert apply_review(str(m), str(t)) == 1
    entries = {e["id"]: e for e in json.load(open(m))}
    assert entries["c1_000"]["verified"] is True
    assert entries["c1_000"]["ref"] == "hello this is Sushant"
    assert entries["c1_000"]["keyterms"] == ["Sushant", "555 1234"]
    assert entries["c2_000"]["verified"] is False and entries["c2_000"]["ref"] == "draft two"


def test_apply_review_unknown_id_raises_and_leaves_manifest_unchanged(tmp_path):
    m, t = _manifest(tmp_path), tmp_path / "review.tsv"
    before = m.read_text()
    t.write_text("id\tcall_id\tdur_s\tdraft\tverified_text\tkeyterms\n"
                 "c1_000\tc1\t3\td\tgood\t\nnope_9\tc1\t3\td\tbad\t\n")
    with pytest.raises(ValueError, match="nope_9"):
        apply_review(str(m), str(t))
    assert m.read_text() == before


def test_export_skips_verified_rows(tmp_path):
    m, t = _manifest(tmp_path), tmp_path / "review.tsv"
    entries = json.load(open(m))
    entries[0]["verified"] = True
    m.write_text(json.dumps(entries))
    export_review(str(m), str(t))
    assert [r["id"] for r in csv.DictReader(open(t), delimiter="\t")] == ["c2_000"]


def test_apply_review_accepts_bom(tmp_path):
    m, t = _manifest(tmp_path), tmp_path / "review.tsv"
    t.write_bytes("\ufeffid\tcall_id\tdur_s\tdraft\tverified_text\tkeyterms\nc1_000\tc1\t3\td\tok\t\n".encode())
    assert apply_review(str(m), str(t)) == 1
