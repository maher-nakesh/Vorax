from zone import ZoneManager


class ThiefDetector:
    def __init__(
        self,
        package_class_id: int = 0,
        person_class_id: int = 1,
        package_ready_frames: int = 30,
        package_missing_frames: int = 15,
        package_move_thresh_px: int = 35,
        alert_cooldown_frames: int = 300,
        package_persist_frames: int = 10,
        delivery_stationary_frames: int = 90,
        delivery_stationary_move_thresh_px: int = 20,
        snatch_window_frames: int = 45,
        snatch_missing_frames: int = 5,
    ):
        self.package_class_id = int(package_class_id)
        self.person_class_id = int(person_class_id)
        self.package_ready_frames = max(1, int(package_ready_frames))
        self.package_missing_frames = max(1, int(package_missing_frames))
        self.package_move_thresh_px = max(0, int(package_move_thresh_px))
        self.alert_cooldown_frames = max(0, int(alert_cooldown_frames))
        self.package_persist_frames = max(0, int(package_persist_frames))
        self.delivery_stationary_frames = max(
            1, int(delivery_stationary_frames)
        )
        self.delivery_stationary_move_thresh_px = max(
            0, int(delivery_stationary_move_thresh_px)
        )
        self.snatch_window_frames = max(0, int(snatch_window_frames))
        self.snatch_missing_frames = max(1, int(snatch_missing_frames))

        self._reset_all_state()

    def _reset_frame_state(self, frame_index: int):
        self.frame_index = int(frame_index)
        self.package_present = False
        self.person_in_zone = False
        self.package_best_center = None
        self.package_best_conf = 0.0
        self.package_best_track_id = None
        self.alert_delivery = False
        self.alert_delivery_reason = None
        self.delivery_driver_track_ids = set()
        self._log_lines = []
        self.person_track_ids_in_zone = set()

    def _reset_all_state(self):
        self.package_seen_frames = 0
        self.package_ready = False
        self.last_package_seen_frame = -10**9
        self.package_missing_count = 0

        self.armed_package_center = None
        self.package_center_history = []

        self.thief_person_track_ids = set()
        self.delivery_person_track_ids = set()
        self.delivery_candidate_track_ids = set()

        self._person_last_center_by_track_id = {}
        self._person_last_seen_frame_by_track_id = {}
        self._person_stationary_frames_by_track_id = {}

        self._prev_package_actually_present = False
        self._package_placed_frame = -10**9
        self.package_placed_unattended = False

        self._last_non_delivery_in_zone_frame = -10**9
        self._last_non_delivery_track_ids_in_zone = set()

        self.last_delivery_alert_frame = -10**9
        self.delivery_received = False

        self.alert_thief = False
        self.alert_reason = None
        self.last_alert_frame = -10**9

        self._current_status = "IDLE"
        self._package_last_known_center = None
        self._package_last_known_frame = -10**9
        self._package_persist_active = False

        self._reset_frame_state(frame_index=0)

    @staticmethod
    def box_center(x1, y1, x2, y2):
        return (x1 + x2) // 2, (y1 + y2) // 2

    def reset(self):
        self._reset_all_state()

    def cleanup_stale_packages(self, frame_index: int):
        self.frame_index = frame_index

    def begin_frame(self, frame_index: int):
        self._reset_frame_state(frame_index=frame_index)

    @property
    def package_in_zone(self) -> bool:
        return bool(self.package_present or self._check_package_persisted())

    def is_delivery_candidate_track_id(self, track_id) -> bool:
        if track_id is None:
            return False
        return int(track_id) in self.delivery_candidate_track_ids

    def _pick_closest_track_id(self, track_ids: set[int], target_center):
        if not track_ids:
            return None
        if target_center is None:
            return int(min(track_ids))

        best_tid = None
        best_dist = float("inf")
        for tid in track_ids:
            center = self._person_last_center_by_track_id.get(int(tid))
            dist = self._dist(center, target_center)
            if dist < best_dist:
                best_dist = dist
                best_tid = int(tid)

        if best_tid is None:
            best_tid = int(min(track_ids))
        return int(best_tid)

    def _get_package_effective_center(self):
        return self.package_best_center or self._package_last_known_center

    def _get_non_delivery_in_zone(self) -> set[int]:
        return set(
            self.person_track_ids_in_zone - self.delivery_person_track_ids
        )

    def _choose_suspects(self, track_ids: set[int]) -> set[int]:
        stationary = {
            tid for tid in track_ids if self._is_track_stationary(tid)
        }
        return set(stationary or track_ids)

    def _set_log_status(self, status: str):
        self._current_status = status
        self._log_lines = [
            f"frame={int(self.frame_index)}",
            f"status={status}",
        ]

    def _trigger_thief_alert(
        self,
        reason: str,
        suspect_track_ids: set[int] | None,
    ):
        self.alert_thief = True
        self.alert_reason = str(reason)
        self.last_alert_frame = int(self.frame_index)

        if suspect_track_ids:
            self.thief_person_track_ids |= {
                int(tid) for tid in set(suspect_track_ids)
            }

        self._set_log_status(f"ALARM ({self.alert_reason})")

    def _trigger_delivery_alert(self, delivery_track_id: int):
        chosen_tid = int(delivery_track_id)
        self.delivery_driver_track_ids = {chosen_tid}
        self.delivery_person_track_ids.add(chosen_tid)
        self.delivery_candidate_track_ids = set()
        self.alert_delivery = True
        self.alert_delivery_reason = "PACKAGE_RECEIVED"
        self.last_delivery_alert_frame = int(self.frame_index)
        self.delivery_received = True

    def update_package_detection(self, center, track_id, conf, zone_rect):
        if zone_rect is None:
            return
        if not ZoneManager.point_in_rect(center, zone_rect):
            return

        self.package_present = True
        self.last_package_seen_frame = self.frame_index
        if conf is None:
            conf = 0.0
        if float(conf) >= float(self.package_best_conf):
            self.package_best_conf = float(conf)
            self.package_best_center = center
            self.package_best_track_id = (
                int(track_id) if track_id is not None else None
            )

        self._package_last_known_center = center
        self._package_last_known_frame = self.frame_index
        self._package_persist_active = True

    def update_person_detection(self, center, track_id, zone_rect):
        if zone_rect is None:
            return False
        if not ZoneManager.point_in_rect(center, zone_rect):
            return False

        self.person_in_zone = True
        if track_id is not None:
            tid = int(track_id)
            self.person_track_ids_in_zone.add(tid)

            last_seen = self._person_last_seen_frame_by_track_id.get(tid)
            last_center = self._person_last_center_by_track_id.get(tid)
            if last_seen is None or last_seen != self.frame_index - 1:
                self._person_stationary_frames_by_track_id[tid] = 0
            else:
                moved = self._dist(center, last_center)
                if moved <= float(self.delivery_stationary_move_thresh_px):
                    self._person_stationary_frames_by_track_id[tid] = (
                        int(
                            self._person_stationary_frames_by_track_id.get(
                                tid, 0
                            )
                        )
                        + 1
                    )
                else:
                    self._person_stationary_frames_by_track_id[tid] = 0

            self._person_last_center_by_track_id[tid] = center
            self._person_last_seen_frame_by_track_id[tid] = self.frame_index
        return True

    def _cleanup_stale_person_tracks(self):
        stale = [
            tid
            for tid, last_seen in (
                self._person_last_seen_frame_by_track_id.items()
            )
            if self.frame_index - int(last_seen) > 2
        ]
        for tid in stale:
            self._person_last_seen_frame_by_track_id.pop(tid, None)
            self._person_last_center_by_track_id.pop(tid, None)
            self._person_stationary_frames_by_track_id.pop(tid, None)

    def _is_track_stationary(self, track_id: int) -> bool:
        if track_id is None:
            return False
        needed = max(1, int(self.delivery_stationary_frames))
        have = int(
            self._person_stationary_frames_by_track_id.get(
                int(track_id),
                0,
            )
        )
        return have >= needed

    def is_track_stationary(self, track_id) -> bool:
        if track_id is None:
            return False
        return self._is_track_stationary(int(track_id))

    @staticmethod
    def _dist(a, b):
        if a is None or b is None:
            return float("inf")
        ax, ay = a
        bx, by = b
        dx = ax - bx
        dy = ay - by
        return (dx * dx + dy * dy) ** 0.5

    def _check_package_persisted(self):
        if not self._package_persist_active:
            return False
        frames_since_seen = self.frame_index - self._package_last_known_frame
        return frames_since_seen <= self.package_persist_frames

    def end_frame(self):
        self._cleanup_stale_person_tracks()

        package_actually_present = bool(
            self.package_present or self._check_package_persisted()
        )

        if not package_actually_present and not self.delivery_received:
            self.delivery_candidate_track_ids = set()

        if package_actually_present:
            self.package_seen_frames += 1
            self.package_missing_count = 0

            effective_center = self._get_package_effective_center()
            if effective_center is not None:
                self.package_center_history.append(effective_center)
                if (
                    len(self.package_center_history)
                    > self.package_ready_frames
                ):
                    self.package_center_history = self.package_center_history[
                        -self.package_ready_frames:
                    ]

            if (
                not self.package_ready
                and (
                    len(self.package_center_history)
                    >= self.package_ready_frames
                )
            ):
                moved = max(
                    self._dist(self.package_center_history[0], pt)
                    for pt in self.package_center_history
                )
                if moved <= float(self.package_move_thresh_px):
                    self.package_ready = True
                    if (
                        self.armed_package_center is None
                        and effective_center is not None
                    ):
                        self.armed_package_center = effective_center
        else:
            self.package_seen_frames = 0
            self.package_center_history = []
            self.package_missing_count += 1

        if (
            package_actually_present
            and not self._prev_package_actually_present
            and not self.delivery_received
        ):
            self._package_placed_frame = int(self.frame_index)
            if self.person_track_ids_in_zone:
                self.delivery_candidate_track_ids = set(
                    int(tid) for tid in self.person_track_ids_in_zone
                )
                self.package_placed_unattended = False
            else:
                self.delivery_candidate_track_ids = set()
                self.package_placed_unattended = True

        if (
            package_actually_present
            and not self.delivery_received
            and not self.package_placed_unattended
            and not self.delivery_candidate_track_ids
            and self.person_track_ids_in_zone
        ):
            self.delivery_candidate_track_ids = set(
                int(tid) for tid in self.person_track_ids_in_zone
            )

        self._prev_package_actually_present = bool(package_actually_present)

        if self.alert_thief and self.person_track_ids_in_zone:
            candidates = (
                self.person_track_ids_in_zone - self.delivery_person_track_ids
            )
            stationary_candidates = {
                tid for tid in candidates if self._is_track_stationary(tid)
            }
            self.thief_person_track_ids |= stationary_candidates

        non_delivery_in_zone = self._get_non_delivery_in_zone()

        if non_delivery_in_zone:
            self._last_non_delivery_in_zone_frame = int(self.frame_index)
            self._last_non_delivery_track_ids_in_zone = set(
                non_delivery_in_zone
            )

        if (
            self.package_ready
            and not self.delivery_received
            and (
                self.frame_index - self.last_delivery_alert_frame
                >= self.alert_cooldown_frames
            )
        ):
            candidate_ids = set(self.delivery_candidate_track_ids)

            chosen_tid = self._pick_closest_track_id(
                candidate_ids,
                self.armed_package_center or self._package_last_known_center,
            )
            if chosen_tid is not None:
                self._trigger_delivery_alert(int(chosen_tid))

        new_status = "IDLE"

        if self.alert_thief:
            reason = self.alert_reason or "ALARM"
            new_status = f"ALARM ({reason})"
        elif self.alert_delivery:
            new_status = "RECEIVED"
        elif (
            self.frame_index - self.last_alert_frame
            < self.alert_cooldown_frames
        ):
            new_status = "COOLDOWN"
        elif (
            self.package_ready
            and (self.delivery_received or self.package_placed_unattended)
            and bool(non_delivery_in_zone)
            and any(
                self._is_track_stationary(tid) for tid in non_delivery_in_zone
            )
        ):
            new_status = "SUSP"
        elif self.package_ready:
            new_status = "READY"
        elif package_actually_present:
            new_status = (
                f"ARMING {self.package_seen_frames}/"
                f"{self.package_ready_frames}"
            )

        self._current_status = new_status
        self._log_lines = [
            f"frame={int(self.frame_index)}",
            f"status={new_status}",
        ]

        if self.alert_thief:
            return

        if (
            self.frame_index - self.last_alert_frame
            < self.alert_cooldown_frames
        ):
            return

        moved_dist = None
        if (
            self.package_ready
            and package_actually_present
            and self.armed_package_center is not None
            and self._get_package_effective_center() is not None
        ):
            effective_center = self._get_package_effective_center()
            moved_dist = self._dist(
                effective_center,
                self.armed_package_center,
            )
            if (
                moved_dist >= float(self.package_move_thresh_px)
                and (self.delivery_received or self.package_placed_unattended)
                and non_delivery_in_zone
            ):
                self._trigger_thief_alert(
                    "PACKAGE_MOVED",
                    self._choose_suspects(set(non_delivery_in_zone)),
                )
                return

        recent_non_delivery = (
            int(self.frame_index) - int(self._last_non_delivery_in_zone_frame)
            <= int(self.snatch_window_frames)
        )
        snatch_ids = set(self._last_non_delivery_track_ids_in_zone)

        if (
            self.package_ready
            and not package_actually_present
            and bool(non_delivery_in_zone)
            and self.package_missing_count >= self.package_missing_frames
        ):
            if not (self.delivery_received or self.package_placed_unattended):
                return
            self._trigger_thief_alert(
                "PACKAGE_MISSING",
                self._choose_suspects(set(non_delivery_in_zone)),
            )
            return

        if (
            self.package_ready
            and not package_actually_present
            and (self.delivery_received or self.package_placed_unattended)
            and self.package_missing_count >= int(self.snatch_missing_frames)
            and recent_non_delivery
            and snatch_ids
        ):
            self._trigger_thief_alert(
                "PACKAGE_SNATCH",
                set(snatch_ids),
            )
            return

    @property
    def suspicious_now(self):
        return self.person_in_zone and self.package_ready

    def get_log_lines(self):
        return list(self._log_lines)

    @property
    def current_status(self):
        return self._current_status

    def is_thief_track_id(self, track_id):
        if track_id is None:
            return False
        return int(track_id) in self.thief_person_track_ids

    def get_alert_info(self):
        if not self.alert_thief:
            return None
        return {
            "frame": self.frame_index,
            "reason": self.alert_reason,
            "thief_track_ids": list(self.thief_person_track_ids),
        }

    def get_delivery_info(self):
        if not self.alert_delivery:
            return None
        return {
            "frame": self.frame_index,
            "reason": self.alert_delivery_reason,
            "delivery_track_ids": list(self.delivery_driver_track_ids),
        }
