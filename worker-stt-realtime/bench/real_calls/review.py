import csv
import json


def export_review(manifest_path, tsv_path, call_ids=None):
    entries = json.load(open(manifest_path))
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "call_id", "dur_s", "draft", "verified_text", "keyterms"])
        for e in entries:
            if e.get("verified") or (call_ids is not None and e["call_id"] not in call_ids):
                continue
            w.writerow([e["id"], e["call_id"], round(e["end_s"] - e["start_s"], 1), e["ref"], "",
                        ";".join(e.get("keyterms", []))])


def apply_review(manifest_path, tsv_path):
    entries = json.load(open(manifest_path))
    by_id = {e["id"]: e for e in entries}
    n = 0
    with open(tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = (row.get("verified_text") or "").strip()
            if not text:
                continue
            e = by_id[row["id"]]
            e["ref"] = text
            e["verified"] = True
            e["keyterms"] = [t.strip() for t in (row.get("keyterms") or "").split(";") if t.strip()]
            n += 1
    json.dump(entries, open(manifest_path, "w"), indent=1)
    return n
