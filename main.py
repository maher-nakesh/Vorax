from __future__ import annotations

import argparse
from pathlib import Path

from app_runtime import PROJECT_ROOT, load_yaml, resolve_path, run_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "main.yaml"),
    )
    parser.add_argument(
        "--notifier-config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "notifier.yaml"),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--heartbeat-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--heartbeat-interval-frames",
        type=int,
        default=30,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_cfg = load_yaml(Path(args.config))
    notifier_cfg = load_yaml(Path(args.notifier_config))
    heartbeat_file = (
        resolve_path(args.heartbeat_file, PROJECT_ROOT)
        if args.heartbeat_file
        else None
    )
    run_detection(
        main_cfg=main_cfg,
        notifier_cfg=notifier_cfg,
        headless=bool(args.headless),
        source_override=args.source,
        heartbeat_file=heartbeat_file,
        heartbeat_interval_frames=int(args.heartbeat_interval_frames),
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
