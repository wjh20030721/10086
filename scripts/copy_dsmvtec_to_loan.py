#!/usr/bin/env python3
import json
import os
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_IMG_ROOT = ROOT / 'Dataset' / 'MMAD' / 'DS-MVTec'
DEST_DIR = ROOT / 'Dataset' / 'test' / 'loan' / 'bank statement'
ANNOT_SRC = ROOT / 'Annotation' / 'DS-MVTec.json'
ANNOT_DEST = ROOT / 'Annotation' / 'test.json'

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

def list_all_images():
    imgs = []
    # Expected pattern: DS-MVTec/<category>/image/<subtype>/file.ext
    for cat_dir in (p for p in SRC_IMG_ROOT.iterdir() if p.is_dir()):
        image_root = cat_dir / 'image'
        if not image_root.exists():
            continue
        for p in image_root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                # Build DS-MVTec.json key
                rel_to_cat_img = p.relative_to(cat_dir)
                key = f"DS-MVTec/{cat_dir.name}/{rel_to_cat_img.as_posix()}"
                imgs.append((p, key, cat_dir.name))
    return imgs

def unique_dest_name(base_name: str) -> str:
    # Sanitize spaces to underscores to align with existing naming style
    base_name = base_name.replace(' ', '_')
    name, ext = os.path.splitext(base_name)
    candidate = base_name
    i = 1
    while (DEST_DIR / candidate).exists():
        candidate = f"{name}_{i}{ext}"
        i += 1
    return candidate

def main(sample_size: int = 25, seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    if not SRC_IMG_ROOT.exists():
        raise SystemExit(f"Source root not found: {SRC_IMG_ROOT}")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(ANNOT_SRC, 'r', encoding='utf-8') as f:
        annot_src = json.load(f)

    # Load or init destination annotations
    if ANNOT_DEST.exists():
        with open(ANNOT_DEST, 'r', encoding='utf-8') as f:
            try:
                annot_dest = json.load(f)
            except json.JSONDecodeError:
                annot_dest = {}
    else:
        annot_dest = {}

    imgs = list_all_images()
    if len(imgs) < sample_size:
        raise SystemExit(f"Not enough images found: {len(imgs)} < {sample_size}")

    sampled = random.sample(imgs, sample_size)

    copied = []
    for src_path, json_key, category in sampled:
        if json_key not in annot_src:
            # Try alternative key without the category duplication handling
            # But DS-MVTec.json uses keys starting with 'DS-MVTec/...'
            print(f"Warning: key not found in annotations: {json_key}")
            continue
        # Construct new filename to avoid collision: dsmvtec_<category>_<subpath>
        subpath = src_path.relative_to(SRC_IMG_ROOT)
        safe_subpath = "_".join(subpath.parts)  # join all parts to single name
        new_name = unique_dest_name(f"dsmvtec_{safe_subpath}")
        dest_path = DEST_DIR / new_name

        # Copy file
        shutil.copy2(src_path, dest_path)

        # Build new annotation entry
        src_entry = annot_src[json_key]
        new_key = f"bank statement/{new_name}"
        new_entry = dict(src_entry)
        new_entry["image_path"] = new_key

        # Merge into destination annotations
        annot_dest[new_key] = new_entry
        copied.append((src_path, dest_path, json_key, new_key))

    # Save updated annotations
    with open(ANNOT_DEST, 'w', encoding='utf-8') as f:
        json.dump(annot_dest, f, ensure_ascii=False, indent=4)

    print(f"Copied {len(copied)} images to '{DEST_DIR}' and updated '{ANNOT_DEST}'.")
    # Brief listing
    for _, dest_path, _, new_key in copied[:10]:
        print(f" - {dest_path.name} -> {new_key}")

if __name__ == '__main__':
    main()
