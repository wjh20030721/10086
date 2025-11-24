import json
from pathlib import Path


TARGETS = list(range(7, 14))


def is_no_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip().lower()
    return t == "no" or t == "no."


def set_detection_to_no(json_path: Path) -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    changed = 0
    for i in TARGETS:
        key = f"refund/{i}.png"
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        conv = entry.get("conversation", [])
        if not isinstance(conv, list):
            continue
        for item in conv:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "Gray Industry Detection":
                continue
            opts = item.get("Options", {})
            if not isinstance(opts, dict):
                continue
            # find option key whose text means No
            target_key = None
            for opt_key, opt_text in opts.items():
                if is_no_text(opt_text):
                    target_key = opt_key
                    break
            if target_key is None:
                # nothing to set
                continue
            if item.get("Answer") != target_key:
                item["Answer"] = target_key
                changed += 1
    if changed:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return changed


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    j = root / "Annotation" / "test.json"
    cnt = set_detection_to_no(j)
    print(f"Updated {cnt} Gray Industry Detection answers to No in refund/7..13")
