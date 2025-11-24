#!/usr/bin/env python3
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_IMG_ROOT = ROOT / 'Dataset' / 'MMAD' / 'DS-MVTec'
DEST_DIR = ROOT / 'Dataset' / 'test' / 'loan' / 'refund'
ANNOT_SRC = ROOT / 'Annotation' / 'DS-MVTec.json'
ANNOT_DEST = ROOT / 'Annotation' / 'test.json'

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
START_INDEX = 7
COUNT = 7

def list_all_images():
    imgs = []
    for cat_dir in (p for p in SRC_IMG_ROOT.iterdir() if p.is_dir()):
        image_root = cat_dir / 'image'
        if not image_root.exists():
            continue
        for p in image_root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel_to_cat = p.relative_to(cat_dir)
                key = f"DS-MVTec/{cat_dir.name}/{rel_to_cat.as_posix()}"
                imgs.append((p, key))
    return imgs

def next_target_names(n: int):
    # Generate desired sequential names from START_INDEX, skipping any existing files
    names = []
    i = START_INDEX
    while len(names) < n:
        # extension decided per-source later; we only reserve the numeric slot here
        names.append(i)
        i += 1
    return names

def main(seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    imgs = list_all_images()
    if len(imgs) < COUNT:
        raise SystemExit(f"Not enough images found in {SRC_IMG_ROOT}")

    # Randomly sample COUNT images
    sampled = random.sample(imgs, COUNT)
    indices = next_target_names(COUNT)

    # Load annotations
    with open(ANNOT_SRC, 'r', encoding='utf-8') as f:
        annot_src = json.load(f)

    if ANNOT_DEST.exists():
        with open(ANNOT_DEST, 'r', encoding='utf-8') as f:
            try:
                annot_dest = json.load(f)
            except json.JSONDecodeError:
                annot_dest = {}
    else:
        annot_dest = {}

    copied = []
    for (src_path, src_key), idx in zip(sampled, indices):
        if src_key not in annot_src:
            print(f"Warning: missing key in DS-MVTec.json: {src_key}")
            continue
        ext = src_path.suffix.lower()
        dst_name = f"{idx}{ext}"
        dst_path = DEST_DIR / dst_name
        if dst_path.exists():
            raise SystemExit(f"Target exists, abort to avoid overwrite: {dst_path}")

        shutil.copy2(src_path, dst_path)

        new_key = f"refund/{dst_name}"
        entry = dict(annot_src[src_key])
        entry['image_path'] = new_key
        annot_dest[new_key] = entry
        copied.append((src_path, dst_path, new_key))

    with open(ANNOT_DEST, 'w', encoding='utf-8') as f:
        json.dump(annot_dest, f, ensure_ascii=False, indent=4)

    print(f"Copied {len(copied)} images to '{DEST_DIR}' and wrote annotations to '{ANNOT_DEST}'.")
    for _, dst, key in copied:
        print(f" - {dst.name} -> {key}")

if __name__ == '__main__':
    main()
