from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _iter_images_flat(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images path is not a directory: {images_dir}")

    images: list[Path] = []
    for p in images_dir.iterdir():  # IMPORTANT: flat only (no recursion)
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def _safe_rmtree_children(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _copy_or_move(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def split_dataset(
    src_images: Path,
    src_labels: Path,
    dst_root: Path,
    train_ratio: float,
    seed: int,
    move: bool,
    clean: bool,
    create_empty_labels: bool,
    dry_run: bool,
) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1 (exclusive)")

    images = _iter_images_flat(src_images)
    if not images:
        raise RuntimeError(f"No images found in: {src_images} (flat scan)")

    dst_images_train = dst_root / "images" / "train"
    dst_images_val = dst_root / "images" / "val"
    dst_labels_train = dst_root / "labels" / "train"
    dst_labels_val = dst_root / "labels" / "val"

    for d in [dst_images_train, dst_images_val, dst_labels_train, dst_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    if clean and not dry_run:
        _safe_rmtree_children(dst_images_train)
        _safe_rmtree_children(dst_images_val)
        _safe_rmtree_children(dst_labels_train)
        _safe_rmtree_children(dst_labels_val)

    rng = random.Random(seed)
    rng.shuffle(images)

    split_idx = int(train_ratio * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    missing_labels = 0

    def handle_one(img: Path, img_dst_dir: Path, lbl_dst_dir: Path) -> None:
        nonlocal missing_labels

        label_src = src_labels / f"{img.stem}.txt"
        img_dst = img_dst_dir / img.name
        lbl_dst = lbl_dst_dir / f"{img.stem}.txt"

        if dry_run:
            return

        _copy_or_move(img, img_dst, move=move)

        if label_src.exists() and label_src.is_file():
            _copy_or_move(label_src, lbl_dst, move=move)
        else:
            missing_labels += 1
            if create_empty_labels:
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                lbl_dst.write_text("", encoding="utf-8")

    for img in train_images:
        handle_one(img, dst_images_train, dst_labels_train)
    for img in val_images:
        handle_one(img, dst_images_val, dst_labels_val)

    print("✅ Dataset split complete")
    print(f"- src_images: {src_images}")
    print(f"- src_labels: {src_labels}")
    print(f"- dst_root:   {dst_root}")
    print(f"- total:      {len(images)}")
    print(f"- train:      {len(train_images)}")
    print(f"- val:        {len(val_images)}")
    print(f"- missing labels (created empty={create_empty_labels}): {missing_labels}")
    if dry_run:
        print("- dry-run:    no files were changed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split YOLO dataset into train/val.\n"
            "IMPORTANT: scans only files in the *root* of src images/labels dirs (no recursion),\n"
            "so existing dataset/images/train and dataset/images/val won't be re-split."
        )
    )
    parser.add_argument("--src-images", default="dataset/images", help="Directory with images (flat)")
    parser.add_argument("--src-labels", default="dataset/labels", help="Directory with YOLO .txt labels (flat)")
    parser.add_argument("--dst", default="dataset", help="Destination dataset root (will write images/train|val, labels/train|val)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio, e.g. 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--move", action="store_true", help="Move files instead of copy")
    parser.add_argument("--clean", action="store_true", help="Clean dst train/val folders before writing")
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        help="Do not create empty .txt label files when label is missing",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show summary but don't write/move anything")

    args = parser.parse_args()

    split_dataset(
        src_images=Path(args.src_images),
        src_labels=Path(args.src_labels),
        dst_root=Path(args.dst),
        train_ratio=args.train_ratio,
        seed=args.seed,
        move=args.move,
        clean=args.clean,
        create_empty_labels=not args.no_empty_labels,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())