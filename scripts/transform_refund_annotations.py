import json
from pathlib import Path


def transform_refund_annotations(
    json_path: Path,
    targets: list[int],
):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    keep_types = {
        "Anomaly Detection": "Gray Industry Detection",
        "Defect Description": "Gray Industry Description",
        "Defect Analysis": "Gray Industry Analysis",
    }

    def strip_extra_fields(obj: dict):
        for k in ["mask_path", "similar_templates", "random_templates"]:
            if k in obj:
                del obj[k]

    changed = 0
    for n in targets:
        key = f"refund/{n}.png"
        if key not in data:
            # also try strict path without slash inconsistencies
            # but keep it silent if not found
            continue

        entry = data[key]
        # strip top-level extra fields
        if isinstance(entry, dict):
            strip_extra_fields(entry)

        conv = entry.get("conversation", []) or []
        # strip extra fields inside conversation and filter types
        new_conv = []
        for item in conv:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in keep_types:
                # remove extra fields but keep Q/A/Options untouched
                strip_extra_fields(item)
                # map type
                item["type"] = keep_types[t]
                new_conv.append(item)

        # replace conversation
        entry["conversation"] = new_conv
        changed += 1

    if changed:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    json_path = root / "Annotation" / "test.json"
    # refund images 7..13
    targets = list(range(7, 14))
    transform_refund_annotations(json_path, targets)
    print(f"Transformed refund annotations for: {targets} in {json_path}")
