# zone.py
import json
from pathlib import Path

import cv2


class ZoneManager:
    def __init__(self, window_name: str, save_path: Path, video_name: str):
        self.window_name = window_name
        self.save_path = Path(save_path)
        self.video_name = Path(video_name).stem
        self.drawing = False
        self.defined = False
        self.start = None
        self.end = None
        self.overlay_alpha = 0.08
        self._zones_data = {}

    def load(self):
        if not self.save_path.exists():
            return
        
        try:
            self._zones_data = json.loads(self.save_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._zones_data = {}
            return

        zones = self._zones_data.get("zones", {})
        video_zone = zones.get(self.video_name)
        
        if video_zone:
            rect = video_zone.get("rect")
            if rect and len(rect) == 4:
                left, top, right, bottom = [int(v) for v in rect]
                self.start = (left, top)
                self.end = (right, bottom)
                self.defined = True
                self.drawing = False

    def save(self):
        rect = self.get_rect(only_if_defined=True)
        if rect is None:
            return

        if not self.save_path.exists():
            self._zones_data = {"zones": {}}
        else:
            try:
                self._zones_data = json.loads(self.save_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._zones_data = {"zones": {}}

        if "zones" not in self._zones_data:
            self._zones_data["zones"] = {}

        existing_zones = self._zones_data["zones"]
        next_id = 1
        for zone_info in existing_zones.values():
            if isinstance(zone_info, dict) and "id" in zone_info:
                next_id = max(next_id, zone_info["id"] + 1)

        if self.video_name in existing_zones:
            next_id = existing_zones[self.video_name].get("id", next_id)

        self._zones_data["zones"][self.video_name] = {
            "rect": list(rect),
            "id": next_id
        }

        self.save_path.write_text(
            json.dumps(self._zones_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def clear(self):
        self.drawing = False
        self.defined = False
        self.start = None
        self.end = None
        
        if self.save_path.exists():
            try:
                data = json.loads(self.save_path.read_text(encoding="utf-8"))
                zones = data.get("zones", {})
                if self.video_name in zones:
                    del zones[self.video_name]
                    data["zones"] = zones
                    self.save_path.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                pass

    def attach(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse, None)

    @staticmethod
    def _normalize_rect(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        return left, top, right, bottom

    def get_rect(self, only_if_defined: bool = False):
        if only_if_defined and not self.defined:
            return None
        if self.start is None or self.end is None:
            return None
        return self._normalize_rect(self.start, self.end)

    @staticmethod
    def point_in_rect(point, rect):
        x, y = point
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def draw(self, frame):
        rect = self.get_rect(only_if_defined=False)
        if rect is None:
            return

        left, top, right, bottom = rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), -1)
        cv2.addWeighted(
            overlay,
            float(self.overlay_alpha),
            frame,
            float(1.0 - self.overlay_alpha),
            0.0,
            frame,
        )

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.defined = False
            self.start = (x, y)
            self.end = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.defined = True
            self.end = (x, y)
            self.save()
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.clear()