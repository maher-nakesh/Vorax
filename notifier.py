from __future__ import annotations

import logging

import requests
import cv2
from datetime import datetime
from pathlib import Path
from typing import Optional


class TelegramNotifier:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        temp_dir: Optional[Path] = None,
        logger: logging.Logger | None = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.temp_dir = (
            Path(temp_dir) if temp_dir else Path("logs") / "alert_screenshots"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._last_alert_time = 0
        self.cooldown_seconds = 60
        self._last_delivery_time = 0
        self.delivery_cooldown_seconds = 60

        self._capturing = False
        self._capture_start_frame = 0
        self._screenshot_count = 0
        self._max_screenshots = 4
        self._delay_frames = [0, 22, 45, 75]
        self._captured_frames = set()
        self._current_status = "IDLE"
        self._capture_caption_prefix = "🚨 ALERT - Screenshot"

        self._send_completion_message = True
        self._completion_message_text = (
            "✅ All 4 screenshots sent. Check camera feed."
        )

    def configure_screenshots(
        self,
        *,
        temp_dir: Optional[Path] = None,
        max_screenshots: Optional[int] = None,
        delay_frames: Optional[list[int]] = None,
        send_completion_message: Optional[bool] = None,
        completion_message_text: Optional[str] = None,
    ) -> None:
        if temp_dir is not None:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        if max_screenshots is not None:
            self._max_screenshots = int(max_screenshots)

        if delay_frames is not None:
            self._delay_frames = [int(v) for v in delay_frames]

        if send_completion_message is not None:
            self._send_completion_message = bool(send_completion_message)

        if completion_message_text is not None:
            self._completion_message_text = str(completion_message_text)

        if (
            self._delay_frames
            and self._max_screenshots
            and len(self._delay_frames) != int(self._max_screenshots)
        ):
            self._logger.warning(
                "Notifier screenshot config mismatch: "
                "max_screenshots=%s delay_frames_len=%s",
                self._max_screenshots,
                len(self._delay_frames),
            )

    def _send_message(self, text: str):
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code != 200:
                self._logger.error(
                    "Telegram sendMessage failed: status=%s response=%s",
                    response.status_code,
                    response.text,
                )
            else:
                self._logger.info("Telegram sendMessage succeeded")
            return response.status_code == 200
        except Exception:
            self._logger.exception("Telegram sendMessage raised exception")
            return False

    def _send_photo(self, photo_path: Path, caption: str = ""):
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption
                }
                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=30,
                )
            if response.status_code != 200:
                self._logger.error(
                    "Telegram sendPhoto failed: status=%s response=%s",
                    response.status_code,
                    response.text,
                )
            else:
                self._logger.info(
                    "Telegram sendPhoto succeeded: path=%s",
                    photo_path,
                )
            return response.status_code == 200
        except Exception:
            self._logger.exception("Telegram sendPhoto raised exception")
            return False

    def start_capture(
        self,
        frame_index: int,
        *,
        caption_prefix: str | None = None,
    ):
        self._capturing = True
        self._capture_start_frame = frame_index
        self._screenshot_count = 0
        self._captured_frames = set()
        self._current_status = "SENDING ALERT..."
        if caption_prefix is not None:
            self._capture_caption_prefix = str(caption_prefix)
        self._logger.info("Notifier capture started: frame=%s", frame_index)

    def is_capturing(self):
        return self._capturing

    def get_capture_status(self):
        return self._current_status

    def try_capture(self, frame, frame_index: int):
        if not self._capturing:
            return None, False

        elapsed = frame_index - self._capture_start_frame

        for i, delay in enumerate(self._delay_frames):
            if elapsed >= delay and i not in self._captured_frames:
                self._captured_frames.add(i)
                self._screenshot_count = len(self._captured_frames)

                timestamp = datetime.now().strftime("%H%M%S")
                filename = (
                    f"alert_{self._screenshot_count}_"
                    f"{self._max_screenshots}_frame{frame_index}_"
                    f"{timestamp}.jpg"
                )
                filepath = self.temp_dir / filename
                wrote = cv2.imwrite(str(filepath), frame)
                if not wrote:
                    self._logger.error(
                        "Failed to write screenshot: path=%s frame=%s",
                        filepath,
                        frame_index,
                    )
                else:
                    self._logger.info(
                        "Screenshot captured: path=%s frame=%s (%s/%s)",
                        filepath,
                        frame_index,
                        self._screenshot_count,
                        self._max_screenshots,
                    )

                self._current_status = (
                    f"SCREENSHOT {self._screenshot_count}/"
                    f"{self._max_screenshots}"
                )

                sent = self._send_photo(
                    filepath,
                    (
                        f"{self._capture_caption_prefix} "
                        f"{self._screenshot_count}/"
                        f"{self._max_screenshots}"
                    ),
                )
                if sent:
                    self._logger.info(
                        "Screenshot sent: path=%s frame=%s (%s/%s)",
                        filepath,
                        frame_index,
                        self._screenshot_count,
                        self._max_screenshots,
                    )
                else:
                    self._logger.error(
                        "Screenshot send failed: path=%s frame=%s (%s/%s)",
                        filepath,
                        frame_index,
                        self._screenshot_count,
                        self._max_screenshots,
                    )

                is_done = self._screenshot_count >= self._max_screenshots
                if is_done:
                    self._capturing = False
                    self._current_status = "ALERT SENT"
                    self._logger.info(
                        "Notifier capture finished: frame=%s",
                        frame_index,
                    )
                    if self._send_completion_message:
                        self._send_message(self._completion_message_text)

                return filepath, is_done

        return None, False

    def send_alert(
        self,
        frame,
        frame_index: int,
        reason: str,
        thief_info: str = "",
    ):
        import time
        current_time = time.time()
        if current_time - self._last_alert_time < self.cooldown_seconds:
            self._logger.info(
                "Alert skipped due to cooldown: frame=%s cooldown_s=%s",
                frame_index,
                self.cooldown_seconds,
            )
            return False

        self._last_alert_time = current_time

        alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"🚨 <b>PORCH PIRATE ALERT</b> 🚨\n"
            f"Time: {alert_time}\n"
            f"Reason: {reason}\n"
            f"Frame: {frame_index}\n"
            f"{thief_info}\n"
            f"Sending 4 screenshots..."
        )
        self._logger.warning(
            "Alarm triggered: reason=%s frame=%s thief_info=%s",
            reason,
            frame_index,
            thief_info,
        )
        ok = self._send_message(message)
        if not ok:
            self._logger.error("Failed to send alert message")
        self.start_capture(
            frame_index,
            caption_prefix="🚨 ALERT - Screenshot",
        )

        return True

    def send_package_received(
        self,
        frame,
        frame_index: int,
        delivery_ids: list[int] | None = None,
    ):
        import time

        current_time = time.time()
        if (
            current_time - self._last_delivery_time
            < self.delivery_cooldown_seconds
        ):
            self._logger.info(
                "Package-received alert skipped due to cooldown: "
                "frame=%s cooldown_s=%s",
                frame_index,
                self.delivery_cooldown_seconds,
            )
            return False

        self._last_delivery_time = current_time

        alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if delivery_ids:
            ids_text = f"Delivery IDs: {delivery_ids}"
        else:
            ids_text = "Delivery IDs: N/A"
        message = (
            f"📦 <b>PACKAGE RECEIVED</b> 📦\n"
            f"Time: {alert_time}\n"
            f"Frame: {frame_index}\n"
            f"{ids_text}\n"
            f"Sending 4 screenshots..."
        )
        self._logger.warning(
            "Package received: frame=%s delivery_ids=%s",
            frame_index,
            delivery_ids,
        )

        ok = self._send_message(message)
        if not ok:
            self._logger.error("Failed to send package-received message")

        self.start_capture(
            frame_index,
            caption_prefix="📦 PACKAGE - Screenshot",
        )
        return True

    def send_test_message(self):
        self._logger.info("Sending notifier test message")
        return self._send_message(
            "✅ Porch Pirate Alert system is active and monitoring."
        )
