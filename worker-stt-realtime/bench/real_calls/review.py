import csv
import json
import os
import tempfile


def _load(path):
    with open(path) as f:
        return json.load(f)


def _write_atomic(path, entries):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(entries, f, indent=1)
    os.replace(tmp, path)


def export_review(manifest_path, tsv_path, call_ids=None):
    entries = _load(manifest_path)
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "call_id", "dur_s", "draft", "verified_text", "keyterms"])
        for e in entries:
            if e.get("verified") or (call_ids is not None and e["call_id"] not in call_ids):
                continue
            w.writerow([e["id"], e["call_id"], round(e["end_s"] - e["start_s"], 1), e["ref"], "",
                        ";".join(e.get("keyterms", []))])


def apply_review(manifest_path, tsv_path):
    entries = _load(manifest_path)
    by_id = {e["id"]: e for e in entries}
    updates = []
    with open(tsv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = (row.get("verified_text") or "").strip()
            if not text:
                continue
            if row["id"] not in by_id:
                raise ValueError(f"review row id {row['id']!r} not found in {manifest_path}")
            terms = [t.strip() for t in (row.get("keyterms") or "").split(";") if t.strip()]
            updates.append((row["id"], text, terms))
    for id_, text, terms in updates:
        e = by_id[id_]
        e["ref"], e["verified"], e["keyterms"] = text, True, terms
    _write_atomic(manifest_path, entries)
    return len(updates)
