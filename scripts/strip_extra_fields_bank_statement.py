#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_JSON = ROOT / 'Annotation' / 'test.json'

STRIP_FIELDS = ['mask_path', 'similar_templates', 'random_templates']

def main():
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changed = 0
    for i in range(25, 50):
        key = f"bank statement/{i}.png"
        if key not in data:
            continue
        entry = data[key]
        before = set(entry.keys())
        for fld in STRIP_FIELDS:
            if fld in entry:
                del entry[fld]
        after = set(entry.keys())
        if before != after:
            changed += 1

    with open(TEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated {changed} entries; removed fields {STRIP_FIELDS} where present.")

if __name__ == '__main__':
    main()
