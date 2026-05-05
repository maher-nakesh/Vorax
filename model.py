# model.py
from datetime import datetime
from shutil import copy2
from pathlib import Path

import yaml

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_paths(value: str | list[str], base_dir: Path) -> list[Path]:
    if isinstance(value, list):
        return [resolve_path(item, base_dir) for item in value]
    return [resolve_path(value, base_dir)]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def _backup_if_exists(path: Path, backups_dir: Path) -> None:
    if not path.exists():
        return
    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backups_dir / (
        f"{path.stem}__backup__{timestamp}{path.suffix}"
    )
    copy2(path, backup_path)
    log(f"Backed up existing weights to: {backup_path}")


def main() -> None:
    config = load_config()

    data_yaml = resolve_path(config["data_yaml"], PROJECT_ROOT)
    weights_path = resolve_path(config["base_weights"], PROJECT_ROOT)
    train_runs_dir = resolve_path(config["project"], PROJECT_ROOT)
    train_run_name = config["name"]
    models_dir = resolve_path(config["models_dir"], PROJECT_ROOT)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    if not weights_path.exists():
        raise FileNotFoundError(
            "Base weights file not found. "
            "Ultralytics may try to download a model when a weights path is "
            "missing. "
            "Set base_weights in "
            f"{CONFIG_PATH} "
            "to an existing local .pt file. "
            f"Missing: {weights_path}"
        )

    dataset_spec = load_yaml(data_yaml)

    train_images = resolve_paths(dataset_spec["train"], data_yaml.parent)
    val_images = resolve_paths(dataset_spec["val"], data_yaml.parent)
    test_images = (
        resolve_paths(dataset_spec["test"], data_yaml.parent)
        if "test" in dataset_spec
        else []
    )

    all_paths = [*train_images, *val_images, *test_images]

    if "train_labels" in dataset_spec:
        all_paths.extend(
            resolve_paths(dataset_spec["train_labels"], data_yaml.parent)
        )
    if "val_labels" in dataset_spec:
        all_paths.extend(
            resolve_paths(dataset_spec["val_labels"], data_yaml.parent)
        )
    if "test_labels" in dataset_spec:
        all_paths.extend(
            resolve_paths(dataset_spec["test_labels"], data_yaml.parent)
        )

    for path in all_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

    log(f"Using dataset YAML: {data_yaml}")
    log(f"Train images: {train_images}")
    log(f"Val images: {val_images}")
    log(f"Test images: {test_images}")

    log(f"Loading model weights: {weights_path}")
    model = YOLO(str(weights_path))

    models_dir.mkdir(parents=True, exist_ok=True)

    # Build training arguments from config
    train_args = {
        "data": str(data_yaml),
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "workers": config["workers"],
        "project": str(train_runs_dir),
        "name": train_run_name,
        "exist_ok": config["exist_ok"],

        # Augmentation
        "augment": config.get("augment", True),
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
        "degrees": config.get("degrees", 0.0),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.5),
        "shear": config.get("shear", 0.0),
        "perspective": config.get("perspective", 0.0),
        "flipud": config.get("flipud", 0.0),
        "fliplr": config.get("fliplr", 0.5),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.0),
        "copy_paste": config.get("copy_paste", 0.0),

        # Regularization
        "patience": config.get("patience", 50),
        "dropout": config.get("dropout", 0.0),
        "weight_decay": config.get("weight_decay", 0.0005),
        "label_smoothing": config.get("label_smoothing", 0.0),

        # Learning rate
        "lr0": config.get("lr0", 0.01),
        "lrf": config.get("lrf", 0.01),
        "momentum": config.get("momentum", 0.937),
        "warmup_epochs": config.get("warmup_epochs", 3.0),
        "warmup_momentum": config.get("warmup_momentum", 0.8),

        # Validation & logging
        "val": config.get("val", True),
        "save": config.get("save", True),
        "save_period": config.get("save_period", -1),
        "verbose": config.get("verbose", True),
        "seed": config.get("seed", 42),
    }

    log(f"Starting training with {config['epochs']} epochs...")
    log(
        "Augmentation: "
        f"HSV({train_args['hsv_h']},"
        f"{train_args['hsv_s']},"
        f"{train_args['hsv_v']}), "
        f"Rotation±{train_args['degrees']}°, "
        f"Scale±{train_args['scale']*100}%, "
        f"Mosaic={train_args['mosaic']}"
    )
    log(
        "Regularization: "
        f"Patience={train_args['patience']}, "
        f"Dropout={train_args['dropout']}, "
        f"WeightDecay={train_args['weight_decay']}"
    )

    model.train(**train_args)

    save_dir = train_runs_dir / train_run_name
    best_src = save_dir / "weights" / "best.pt"
    last_src = save_dir / "weights" / "last.pt"
    best_dst = models_dir / config["best_output"]
    last_dst = models_dir / config["last_output"]
    backups_dir = models_dir / "backups"

    if best_src.exists():
        _backup_if_exists(best_dst, backups_dir)
        copy2(best_src, best_dst)
        log(f"Copied best weights to: {best_dst}")
    if last_src.exists():
        _backup_if_exists(last_dst, backups_dir)
        copy2(last_src, last_dst)
        log(f"Copied last weights to: {last_dst}")

    # Log training results
    results_file = save_dir / "results.csv"
    if results_file.exists():
        log(f"Training results saved: {results_file}")

    log(f"Training complete. Results saved in: {save_dir}")


if __name__ == "__main__":
    main()
