import json
from pathlib import Path


def transform_repayment_annotations(json_path: Path, targets: list[int]):
    data = json.loads(json_path.read_text(encoding="utf-8"))

    keep_types = {
        "Anomaly Detection": "Gray Industry Detection",
        "Defect Description": "Gray Industry Description",
        "Defect Analysis": "Gray Industry Analysis",
    }

    def strip_extra_fields(obj: dict):
        for k in ("mask_path", "similar_templates", "random_templates"):
            if k in obj:
                del obj[k]

    for n in targets:
        key = f"repayment/{n}.png"
        entry = data.get(key)
        if not isinstance(entry, dict):
            continue
        strip_extra_fields(entry)
        conv = entry.get("conversation", []) or []
        new_conv = []
        for item in conv:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in keep_types:
                strip_extra_fields(item)
                item["type"] = keep_types[t]
                new_conv.append(item)
        entry["conversation"] = new_conv

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    j = root / "Annotation" / "test.json"
    transform_repayment_annotations(j, list(range(8, 16)))
    print("Transformed repayment annotations for 8..15")
