#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_JSON = ROOT / 'Annotation' / 'test.json'

TARGET_KEYS = [f"bank statement/{i}.png" for i in range(25,50)]

def is_no_text(v: str) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip().lower().rstrip('.')
    return s == 'no'

def main():
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = 0
    for key in TARGET_KEYS:
        entry = data.get(key)
        if not entry:
            continue
        conv = entry.get('conversation')
        if not isinstance(conv, list):
            continue
        # find detection item
        for item in conv:
            if item.get('type') == 'Gray Industry Detection':
                opts = item.get('Options', {})
                # prefer an option whose value is exactly No/No.
                no_key = None
                for k, v in opts.items():
                    if is_no_text(v):
                        no_key = k
                        break
                # fallback: if not found, but there is a 'B' and it looks like No variant
                if no_key is None and 'B' in opts and is_no_text(opts.get('B')):
                    no_key = 'B'
                if no_key is not None and item.get('Answer') != no_key:
                    item['Answer'] = no_key
                    updated += 1
                break

    with open(TEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Updated Gray Industry Detection answers to 'No' for {updated} entries.")

if __name__ == '__main__':
    main()
