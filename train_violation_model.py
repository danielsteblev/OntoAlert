"""
Обучение YOLO модели на ваш датасет нарушений (smoking / littering / graffiti).

Пример:
  python train_violation_model.py --data data.yaml --model yolov8n.pt --epochs 50 --device cpu

Результат:
  runs/violation_detection/train/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _find_latest_run_dir(project_dir: Path, base_name: str) -> Path | None:
    """
    Ultralytics may auto-increment run name (train -> train2 -> train3...) if exist_ok=False.
    This finds the newest directory among {base_name, base_name2, base_name3, ...}.
    """
    if not project_dir.exists() or not project_dir.is_dir():
        return None

    candidates: list[Path] = []
    for p in project_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == base_name or (p.name.startswith(base_name) and p.name[len(base_name) :].isdigit()):
            candidates.append(p)

    if not candidates:
        return None

    return max(candidates, key=lambda d: d.stat().st_mtime)


def _load_data_yaml(data_path: Path) -> dict:
    try:
        import yaml  # PyYAML (usually present via ultralytics)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read data.yaml. Install with: python -m pip install pyyaml"
        ) from e

    raw = data_path.read_text(encoding="utf-8", errors="strict")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid data.yaml structure: expected dict, got {type(data)}")
    return data


def _count_label_ids(labels_dir: Path) -> dict[int, int]:
    """
    Counts YOLO class ids in label .txt files (first integer on each non-empty line).
    """
    import re
    from collections import Counter

    c: Counter[int] = Counter()
    if not labels_dir.exists():
        return {}

    for f in labels_dir.glob("*.txt"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s", line)
            if m:
                c[int(m.group(1))] += 1

    return dict(sorted(c.items()))


def _print_dataset_mapping(data_path: Path) -> None:
    resolved = data_path.resolve()
    print("📄 data.yaml file")
    print(f"- path_arg: {data_path}")
    print(f"- abs_path: {resolved}")
    try:
        raw = resolved.read_text(encoding="utf-8", errors="replace")
        raw_preview = "\n".join(raw.splitlines()[:20])
        print("- head(20):")
        for line in raw_preview.splitlines():
            print(f"  {line}")
    except Exception as e:
        print(f"⚠️ Could not read data.yaml for preview: {e}")

    data = _load_data_yaml(resolved)
    names = data.get("names", [])
    if isinstance(names, dict):
        names_map = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        names_map = {i: str(v) for i, v in enumerate(names)}
    else:
        names_map = {}

    print("📄 data.yaml summary")
    print(f"- path:  {data.get('path')}")
    print(f"- train: {data.get('train')}")
    print(f"- val:   {data.get('val')}")
    print(f"- nc:    {data.get('nc')}")
    print(f"- names: {names_map}")

    # Best-effort label id distribution check for the common local layout
    train_counts = _count_label_ids(Path("dataset/labels/train"))
    val_counts = _count_label_ids(Path("dataset/labels/val"))
    if train_counts or val_counts:
        print("🏷️ label id counts (dataset/labels)")
        print(f"- train: {train_counts}")
        print(f"- val:   {val_counts}")

        # Warn if any id is missing in names_map
        all_ids = set(train_counts) | set(val_counts)
        missing = sorted(i for i in all_ids if i not in names_map)
        if missing:
            print(f"⚠️ WARNING: label ids not present in data.yaml names: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train custom YOLO model for violations dataset")
    parser.add_argument("--data", default="data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model weights (e.g. yolov8n.pt, yolov8s.pt) or a checkpoint .pt",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | 0 | 0,1 ...")
    parser.add_argument("--name", default="train", help="Run name inside project dir")
    parser.add_argument("--project", default="violation_detection", help="Project dir for training runs")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (Windows/CPU: keep 0)")
    parser.add_argument(
        "--save-period",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (1 = each epoch; helps to keep partial progress)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the provided --model checkpoint (typically .../weights/last.pt)",
    )

    args = parser.parse_args()

    # Common Windows issue: user accidentally types Cyrillic 'с' instead of Latin 'c' in "cpu"
    # (looks identical, but breaks Ultralytics device parsing).
    raw_device = str(args.device)
    fixed_device = raw_device.replace("с", "c").replace("С", "C")
    if fixed_device != raw_device:
        print(f"⚠️ Detected non-latin device string: '{raw_device}' -> '{fixed_device}'")
    args.device = fixed_device

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    print("🎓 Training violation model")
    print(f"- data:   {data_path}")
    print(f"- model:  {args.model}")
    print(f"- epochs: {args.epochs}")
    print(f"- imgsz:  {args.imgsz}")
    print(f"- batch:  {args.batch}")
    print(f"- device: {args.device}")
    print(f"- out:    {args.project}/{args.name}")
    _print_dataset_mapping(data_path)

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        save_period=args.save_period,
        resume=args.resume,
        save=True,
        plots=True,
    )

    # Find actual run dir (train/train2/...)
    project_dir = Path(args.project)
    run_dir = _find_latest_run_dir(project_dir, args.name) or (project_dir / args.name)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    if best.exists():
        print(f"✅ Done. Best weights: {best}")
        print("💡 Чтобы бот/камера использовали модель, добавьте в .env:")
        print(f"   YOLO_MODEL_PATH={best.as_posix()}")
    elif last.exists():
        print(f"⚠️ best.pt not found, but last.pt exists: {last}")
        print("💡 Можно использовать last.pt (чуть хуже/не гарантированно лучший чекпойнт):")
        print(f"   YOLO_MODEL_PATH={last.as_posix()}")
    else:
        print("⚠️ Training finished, but best.pt not found (check training directory).")
        print(f"   Expected run dir: {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

