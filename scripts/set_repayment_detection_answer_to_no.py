import json
from pathlib import Path


def is_no_text(text: str) -> bool:
    return isinstance(text, str) and text.strip().lower() in {"no", "no."}


def set_detection_to_no(json_path: Path, start: int, end: int, prefix: str = "repayment") -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    changed = 0
    for i in range(start, end + 1):
        key = f"{prefix}/{i}.png"
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        conv = entry.get("conversation", [])
        if not isinstance(conv, list):
            continue
        for item in conv:
            if item.get("type") != "Gray Industry Detection":
                continue
            opts = item.get("Options", {})
            if not isinstance(opts, dict):
                continue
            target = None
            for k, v in opts.items():
                if is_no_text(v):
                    target = k
                    break
            if target and item.get("Answer") != target:
                item["Answer"] = target
                changed += 1
    if changed:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return changed


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    j = root / "Annotation" / "test.json"
    cnt = set_detection_to_no(j, 8, 15, "repayment")
    print(f"Updated {cnt} Gray Industry Detection answers to No in repayment/8..15")
