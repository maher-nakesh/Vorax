from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from detection import ThiefDetector
from incident_logger import IncidentLogger
from logging_setup import configure_logging
from notifier import TelegramNotifier
from zone import ZoneManager


PROJECT_ROOT = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML (expected mapping): {path}")
    return data


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def as_list(values):
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def draw_status_box(frame, lines: list[str]):
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 8
    line_gap = 6

    widths: list[int] = []
    heights: list[int] = []
    for text in lines:
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        widths.append(int(w))
        heights.append(int(h))

    box_w = max(widths) + padding * 2
    box_h = sum(heights) + padding * 2 + line_gap * (len(lines) - 1)

    x0, y0 = 10, 10
    x1, y1 = x0 + box_w, y0 + box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, frame)

    y = y0 + padding
    for text, h in zip(lines, heights):
        y += h
        cv2.putText(
            frame,
            text,
            (x0 + padding, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap


def _write_heartbeat(path: Path, frame_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"frame={frame_index}\n", encoding="utf-8")


def build_notifier(
    *,
    credentials_path: Path,
    notifier_cfg: dict,
    app_logger: logging.Logger,
    project_root: Path,
) -> TelegramNotifier | None:
    if not credentials_path.exists():
        app_logger.warning(
            "Notifications disabled (credentials not found): %s",
            credentials_path,
        )
        return None

    try:
        creds = json.loads(credentials_path.read_text(encoding="utf-8"))
        bot_token = creds["telegram_bot_token"]
        chat_id = creds["telegram_chat_id"]
    except Exception:
        app_logger.exception(
            "Failed to load credentials; notifications disabled"
        )
        return None

    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        temp_dir=resolve_path(
            notifier_cfg.get("screenshots", {}).get(
                "temp_dir",
                "logs/alert_screenshots",
            ),
            project_root,
        ),
        logger=app_logger.getChild("notifier"),
    )

    screenshots_cfg = notifier_cfg.get("screenshots", {})
    notifier.configure_screenshots(
        temp_dir=resolve_path(
            screenshots_cfg.get("temp_dir", "logs/alert_screenshots"),
            project_root,
        ),
        max_screenshots=screenshots_cfg.get("max_screenshots"),
        delay_frames=screenshots_cfg.get("delay_frames"),
        send_completion_message=screenshots_cfg.get("send_completion_message"),
        completion_message_text=screenshots_cfg.get("completion_message_text"),
    )
    return notifier


def run_detection(
    *,
    main_cfg: dict,
    notifier_cfg: dict,
    headless: bool = False,
    source_override: str | None = None,
    heartbeat_file: Path | None = None,
    heartbeat_interval_frames: int = 30,
    project_root: Path = PROJECT_ROOT,
) -> None:
    log_file = configure_logging(
        log_dir=resolve_path(main_cfg["paths"]["logs_dir"], project_root),
        log_level=str(main_cfg.get("logging", {}).get("level", "INFO")),
        rotate_max_bytes=int(
            main_cfg.get("logging", {}).get("rotate_max_bytes", 5_000_000)
        ),
        rotate_backup_count=int(
            main_cfg.get("logging", {}).get("rotate_backup_count", 5)
        ),
        console=bool(main_cfg.get("logging", {}).get("console", True)),
    )
    app_logger = logging.getLogger("parcel_alert")

    video_name = str(main_cfg["video"]["name"])
    if source_override:
        video_path = resolve_path(source_override, project_root)
        video_name = video_path.name
    else:
        video_path = resolve_path(Path("videos") / video_name, project_root)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    window_name = str(main_cfg["ui"]["window_name"])
    zone_path = resolve_path(main_cfg["paths"]["zone"], project_root)

    yolo_cfg = main_cfg["yolo"]
    weights_primary = resolve_path(yolo_cfg["weights_primary"], project_root)
    weights_fallback = resolve_path(yolo_cfg["weights_fallback"], project_root)
    weights_path = (
        weights_primary if weights_primary.exists() else weights_fallback
    )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_primary} or {weights_fallback}"
        )

    app_logger.info("Detection starting")
    app_logger.info("Log file: %s", log_file)
    app_logger.info("Video: %s", video_path)
    app_logger.info("Weights: %s", weights_path)
    app_logger.info("Zone config: %s", zone_path)

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1.0:
        fps = 30.0

    zone = ZoneManager(window_name, zone_path, video_name)
    zone.load()
    if not headless:
        zone.attach()
    elif not zone.defined:
        raise RuntimeError(
            "Headless mode requires a saved zone for this video. "
            "Run once with UI enabled to draw/save the zone."
        )

    classes_cfg = main_cfg["classes"]
    package_class_id = int(classes_cfg["package_class_id"])
    person_class_id = int(classes_cfg["person_class_id"])

    detector_cfg = main_cfg.get("detector", {})

    package_ready_seconds = detector_cfg.get("package_ready_seconds")
    if package_ready_seconds is not None:
        package_ready_frames = max(
            1,
            int(round(float(package_ready_seconds) * fps)),
        )
    else:
        package_ready_frames = int(
            detector_cfg.get("package_ready_frames", 30)
        )

    delivery_stationary_seconds = float(
        detector_cfg.get("delivery_stationary_seconds", 3.0)
    )
    delivery_stationary_frames = max(
        1,
        int(round(delivery_stationary_seconds * fps)),
    )

    detector = ThiefDetector(
        package_class_id=package_class_id,
        person_class_id=person_class_id,
        package_ready_frames=int(package_ready_frames),
        package_missing_frames=int(
            detector_cfg.get("package_missing_frames", 15)
        ),
        package_move_thresh_px=int(
            detector_cfg.get("package_move_thresh_px", 35)
        ),
        alert_cooldown_frames=int(
            detector_cfg.get("alert_cooldown_frames", 300)
        ),
        package_persist_frames=int(
            detector_cfg.get("package_persist_frames", 10)
        ),
        delivery_stationary_frames=int(delivery_stationary_frames),
        delivery_stationary_move_thresh_px=int(
            detector_cfg.get("delivery_stationary_move_thresh_px", 20)
        ),
    )

    incident_logger = IncidentLogger(
        resolve_path(main_cfg["paths"]["incidents_dir"], project_root),
        logger=app_logger.getChild("incidents"),
    )

    notifier = build_notifier(
        credentials_path=resolve_path(
            main_cfg["paths"]["credentials"],
            project_root,
        ),
        notifier_cfg=notifier_cfg,
        app_logger=app_logger,
        project_root=project_root,
    )

    if notifier and bool(
        main_cfg.get("notifier", {}).get("send_test_message_on_start", True)
    ):
        notifier.send_test_message()

    frame_index = 0
    incident_started = False
    last_notified_alert_frame: int | None = None
    last_notified_delivery_frame: int | None = None

    heartbeat_interval_frames = max(1, int(heartbeat_interval_frames))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                app_logger.info("End of video")
                break

            frame_index += 1

            if heartbeat_file and (
                frame_index % heartbeat_interval_frames == 0
            ):
                try:
                    _write_heartbeat(heartbeat_file, frame_index)
                except Exception:
                    app_logger.exception(
                        "Failed to write heartbeat: %s", heartbeat_file
                    )

            results = model.track(
                frame,
                persist=True,
                conf=float(yolo_cfg["conf_threshold"]),
                iou=float(yolo_cfg["iou_threshold"]),
                imgsz=int(yolo_cfg["img_size"]),
                classes=[int(v) for v in yolo_cfg["target_class_ids"]],
                max_det=int(yolo_cfg["max_det"]),
                vid_stride=int(yolo_cfg.get("vid_stride", 1)),
                device=yolo_cfg.get("device", None),
                half=bool(yolo_cfg.get("half", False)),
                verbose=False,
            )

            result = results[0]
            boxes_obj = result.boxes

            zone.draw(frame)
            zone_rect = zone.get_rect(only_if_defined=True)
            detector.cleanup_stale_packages(frame_index)
            detector.begin_frame(frame_index)

            if zone_rect is not None and not incident_started:
                incident_logger.start_incident(video_name, zone_rect)
                incident_started = True

            if boxes_obj is not None:
                xyxy = as_list(boxes_obj.xyxy)
                class_ids = [int(v) for v in as_list(boxes_obj.cls)]
                confidences = [float(v) for v in as_list(boxes_obj.conf)]
                track_ids = (
                    as_list(boxes_obj.id)
                    if boxes_obj.id is not None
                    else [None] * len(xyxy)
                )

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    track_id = track_ids[i] if i < len(track_ids) else None

                    class_name = model.names.get(class_id, str(class_id))
                    label = f"{class_name} {confidence:.2f}"
                    if track_id is not None:
                        label = (
                            f"{class_name} ID:{int(track_id)} "
                            f"{confidence:.2f}"
                        )

                    center = detector.box_center(x1, y1, x2, y2)

                    if class_id == package_class_id:
                        detector.update_package_detection(
                            center=center,
                            track_id=track_id,
                            conf=confidence,
                            zone_rect=zone_rect,
                        )
                        if zone_rect is not None and not zone.point_in_rect(
                            center,
                            zone_rect,
                        ):
                            continue

                    is_person_in_zone = False
                    is_thief = False
                    is_delivery = False
                    is_delivery_candidate = False

                    if class_id == person_class_id:
                        is_person_in_zone = detector.update_person_detection(
                            center=center,
                            track_id=track_id,
                            zone_rect=zone_rect,
                        )

                        is_delivery_candidate = (
                            detector.is_delivery_candidate_track_id(track_id)
                        )

                        if track_id is not None:
                            is_delivery = (
                                int(track_id)
                                in detector.delivery_person_track_ids
                            )

                        is_thief = detector.is_thief_track_id(track_id)
                        if is_thief:
                            label = f"THIEF {confidence:.2f}"
                        elif is_delivery:
                            label = f"DELIVERY {confidence:.2f}"
                        elif (
                            is_person_in_zone
                            and not is_delivery
                            and (
                                (
                                    detector.package_in_zone
                                    and is_delivery_candidate
                                )
                                or (
                                    detector.package_ready
                                    and (
                                        detector.delivery_received
                                        or detector.package_placed_unattended
                                    )
                                )
                            )
                        ):
                            label = f"SUSP {confidence:.2f}"

                    color = (0, 255, 0)
                    if class_id == person_class_id:
                        if is_thief:
                            color = (0, 0, 255)
                        elif (
                            is_person_in_zone
                            and not is_delivery
                            and (
                                (
                                    detector.package_in_zone
                                    and is_delivery_candidate
                                )
                                or (
                                    detector.package_ready
                                    and (
                                        detector.delivery_received
                                        or detector.package_placed_unattended
                                    )
                                )
                            )
                        ):
                            color = (0, 255, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            detector.end_frame()

            if incident_started:
                details = {}
                alert_info = detector.get_alert_info()
                delivery_info = detector.get_delivery_info()
                if alert_info:
                    details["thief_alert"] = alert_info
                if delivery_info:
                    details["delivery_alert"] = delivery_info

                incident_logger.log_state_change(
                    frame_index,
                    detector.current_status,
                    details or None,
                )

            if notifier and incident_started:
                delivery_info = detector.get_delivery_info() or {}
                delivery_frame = int(delivery_info.get("frame", -1))
                if (
                    delivery_frame >= 0
                    and delivery_frame != last_notified_delivery_frame
                    and not notifier.is_capturing()
                ):
                    delivery_ids = delivery_info.get("delivery_track_ids")
                    ok = notifier.send_package_received(
                        frame=frame,
                        frame_index=frame_index,
                        delivery_ids=list(delivery_ids or []),
                    )
                    app_logger.info(
                        "Delivery dispatch result: ok=%s frame=%s ids=%s",
                        ok,
                        frame_index,
                        delivery_ids,
                    )
                    last_notified_delivery_frame = delivery_frame

            if notifier and incident_started and detector.alert_thief:
                alert_info = detector.get_alert_info() or {}
                alert_frame = int(alert_info.get("frame", -1))
                if (
                    alert_frame >= 0
                    and alert_frame != last_notified_alert_frame
                ):
                    thief_ids = alert_info.get("thief_track_ids", [])
                    reason = detector.alert_reason or str(
                        alert_info.get("reason") or "UNKNOWN"
                    )
                    incident_logger.log_alert(
                        frame_index,
                        reason,
                        list(thief_ids),
                    )
                    ok = notifier.send_alert(
                        frame=frame,
                        frame_index=frame_index,
                        reason=reason,
                        thief_info=(
                            f"Thief IDs: {thief_ids}"
                            if thief_ids
                            else "Unknown"
                        ),
                    )
                    app_logger.info(
                        "Alert dispatch result: ok=%s reason=%s frame=%s",
                        ok,
                        reason,
                        frame_index,
                    )
                    last_notified_alert_frame = alert_frame

            if notifier and notifier.is_capturing():
                _filepath, is_done = notifier.try_capture(frame, frame_index)
                if is_done:
                    app_logger.info(
                        "Notifier capture sequence completed: frame=%s",
                        frame_index,
                    )

            status_lines = detector.get_log_lines()
            if notifier and notifier.is_capturing():
                capture_status = notifier.get_capture_status()
                if capture_status:
                    status_lines.append(capture_status)

            if zone_rect is not None:
                draw_status_box(frame, status_lines)

            if not headless:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(15) & 0xFF == ord("q"):
                    app_logger.info("Quit requested by user")
                    break
    finally:
        incident_logger.end_incident()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        app_logger.info("Detection stopped")
