import json
import logging
from datetime import datetime
from pathlib import Path


class IncidentLogger:
    def __init__(self, log_dir: Path, logger: logging.Logger | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger or logging.getLogger(__name__)
        self.current_incident = None
        self.current_file = None
        self._last_status = None

    def start_incident(self, video_name: str, zone_rect):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_video = Path(video_name).stem.replace(" ", "_")
        incident_id = f"{safe_video}_{timestamp}"

        self.current_incident = {
            "incident_id": incident_id,
            "video_name": video_name,
            "started_at": datetime.now().isoformat(),
            "zone_rect": list(zone_rect) if zone_rect else None,
            "events": [],
        }
        self.current_file = self.log_dir / f"{incident_id}.json"
        self._last_status = "IDLE"
        self._save()
        self._logger.info(
            "Incident started: id=%s video=%s zone_rect=%s",
            incident_id,
            video_name,
            list(zone_rect) if zone_rect else None,
        )

    def log_state_change(
        self,
        frame_index: int,
        new_status: str,
        details: dict | None = None,
    ):
        if self.current_incident is None:
            return False

        if new_status == self._last_status:
            return False  # No change, don't log

        self._last_status = new_status

        event = {
            "frame": frame_index,
            "timestamp": datetime.now().isoformat(),
            "type": "STATE_CHANGE",
            "status": new_status,
        }
        if details:
            event["details"] = details

        self.current_incident["events"].append(event)
        self._save()
        if details:
            self._logger.info(
                "State change: frame=%s status=%s details=%s",
                frame_index,
                new_status,
                details,
            )
        else:
            self._logger.info(
                "State change: frame=%s status=%s",
                frame_index,
                new_status,
            )
        return True

    def log_alert(self, frame_index: int, reason: str, thief_track_ids: list):
        self.log_state_change(
            frame_index,
            "ALARM",
            {
                "reason": reason,
                "thief_track_ids": thief_track_ids,
            }
        )

    def end_incident(self):
        if self.current_incident is None:
            return

        self.current_incident["ended_at"] = datetime.now().isoformat()
        self._save()
        self._logger.info(
            "Incident ended: id=%s",
            self.current_incident.get("incident_id"),
        )
        self.current_incident = None
        self.current_file = None
        self._last_status = None

    def _save(self):
        if self.current_file and self.current_incident:
            self.current_file.write_text(
                json.dumps(
                    self.current_incident,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
