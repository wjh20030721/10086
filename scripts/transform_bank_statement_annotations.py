#!/usr/bin/env python3
"""
Transform Annotation/test.json entries for bank statement/25.png..49.png:
- Keep only three conversation types: Anomaly Detection, Defect Description, Defect Analysis
- Rename their types to: Gray Industry Detection, Gray Industry Description, Gray Industry Analysis
- Preserve question/answer/options/annotation fields untouched
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
TEST_JSON = ROOT / 'Annotation' / 'test.json'

KEEP_ORDER = [
    'Anomaly Detection',
    'Defect Description',
    'Defect Analysis',
]

TYPE_MAP = {
    'Anomaly Detection': 'Gray Industry Detection',
    'Defect Description': 'Gray Industry Description',
    'Defect Analysis': 'Gray Industry Analysis',
}

def transform_entry(entry: dict) -> bool:
    conv = entry.get('conversation')
    if not isinstance(conv, list):
        return False
    # Map of type -> list of items preserving original order
    by_type = {}
    for item in conv:
        t = item.get('type')
        if t in KEEP_ORDER:
            by_type.setdefault(t, []).append(item)

    # Build new conversation in desired order, keeping original items
    new_conv = []
    for t in KEEP_ORDER:
        items = by_type.get(t, [])
        for it in items:
            # Only change the type field
            it = dict(it)
            it['type'] = TYPE_MAP[t]
            new_conv.append(it)

    # Only update if there is any change
    entry['conversation'] = new_conv
    return True

def main():
    if not TEST_JSON.exists():
        raise SystemExit(f"File not found: {TEST_JSON}")

    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = 0
    for i in range(25, 50):
        key = f"bank statement/{i}.png"
        if key in data:
            if transform_entry(data[key]):
                updated += 1

    with open(TEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated {updated} entries in {TEST_JSON}.")

if __name__ == '__main__':
    main()
