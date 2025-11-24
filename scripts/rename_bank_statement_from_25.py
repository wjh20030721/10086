#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = ROOT / 'Dataset' / 'test' / 'loan' / 'bank statement'
TEST_JSON = ROOT / 'Annotation' / 'test.json'

START_INDEX = 25
PREFIX = 'dsmvtec_'

def main():
    if not BANK_DIR.exists():
        raise SystemExit(f"bank statement dir not found: {BANK_DIR}")
    if not TEST_JSON.exists():
        raise SystemExit(f"test.json not found: {TEST_JSON}")

    # Collect files to rename: the newly added dsmvtec_* files
    files = [p for p in BANK_DIR.iterdir() if p.is_file() and p.name.startswith(PREFIX)]
    if not files:
        print("No files to rename (no dsmvtec_* found).")
        return

    # Stable order: by modification time, fallback to name
    files.sort(key=lambda p: (p.stat().st_mtime, p.name))

    # Compute target names 25..(25+N-1)
    targets = []
    for i, p in enumerate(files):
        new_name = f"{START_INDEX + i}{p.suffix.lower()}"
        targets.append((p, p.with_name(new_name)))

    # Safety check: avoid overwriting existing target files
    collisions = [dst for _, dst in targets if dst.exists()]
    if collisions:
        names = ', '.join(d.name for d in collisions[:5])
        raise SystemExit(f"Target files already exist, aborting to avoid overwrite: {names} ...")

    # Load JSON
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply renames on disk and update JSON keys
    moved = []
    for src, dst in targets:
        src_name = src.name
        dst_name = dst.name
        src.rename(dst)

        old_key = f"bank statement/{src_name}"
        new_key = f"bank statement/{dst_name}"
        if old_key in data:
            entry = data.pop(old_key)
            entry['image_path'] = new_key
            data[new_key] = entry
            moved.append((src_name, dst_name, True))
        else:
            # If not found (unexpected), still record move
            moved.append((src_name, dst_name, False))

    # Save JSON
    with open(TEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Renamed {len(moved)} files starting from index {START_INDEX}.")
    missing = [m for m in moved if not m[2]]
    if missing:
        print(f"Warning: {len(missing)} entries not found in JSON and therefore not updated.")
        for s, d, _ in missing[:5]:
            print(f" - Missing JSON key for: bank statement/{s}")
    else:
        print("All corresponding JSON entries updated.")

if __name__ == '__main__':
    main()
