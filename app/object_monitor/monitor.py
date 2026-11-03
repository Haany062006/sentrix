"""
SENTRIX Monitored Object Watcher
Robust multi-modal presence detection (SIFT + RANSAC Homography + Multi-scale Template + ORB).
Maintains exact timestamps and database intervals for when the object was missing (from - to).
NOTE: As per user instructions, NO bounding box is drawn on the camera frame.
"""

import os
import time
from datetime import datetime
import threading
import cv2
import numpy as np

from app.database.db import (
    start_missing_interval,
    end_missing_interval,
    get_missing_intervals
)


class ObjectMonitor:
    """
    Manages user-uploaded object template, multi-modal matching in background,
    and missing object alert state with exact interval tracking.
    """

    GRACE_PERIOD_SECONDS = 1.2
    MIN_MISSES_FOR_ALERT = 4

    def __init__(self, storage_path="data/monitored_object.png"):
        self.storage_path = storage_path
        self.template_bgr = None
        self.template_gray = None
        self.template_w = 0
        self.template_h = 0
        self.object_name = ""
        self.has_object = False

        self.is_present = False
        self.confidence = 0.0
        self.consecutive_misses = 0
        self.consecutive_hits = 0

        # Exact timelines & intervals
        self.last_seen_time = None
        self.missing_start_time = None
        self.active_interval_id = None
        self.alert_newly_triggered = False
        self.restored_newly_triggered = False
        self.GRACE_PERIOD_SECONDS = 0.8
        self.MIN_MISSES_FOR_ALERT = 3

        # SIFT detector (scale & rotation invariant)
        try:
            self.sift = cv2.SIFT_create(nfeatures=800)
            self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            self.has_sift = True
        except Exception:
            self.sift = None
            self.flann = None
            self.has_sift = False

        # ORB detector (fallback / complementary)
        self.orb = cv2.ORB_create(nfeatures=600)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.template_sift_kp = None
        self.template_sift_des = None
        self.template_orb_kp = None
        self.template_orb_des = None

        self.lock = threading.Lock()

        # Background worker state
        self._latest_frame = None
        self._frame_version = 0
        self._processed_version = -1
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="SentrixObjectMonitorThread"
        )
        self._worker_thread.start()

        # Load persisted template if available
        self._load_saved_template()

    def _load_saved_template(self):
        """Load previously saved object image from disk if it exists."""
        if os.path.exists(self.storage_path):
            try:
                img = cv2.imread(self.storage_path)
                if img is not None and img.size > 0:
                    self._setup_template(img, name="Saved Monitored Object")
            except Exception as e:
                print(f"[ObjectMonitor] Failed to load saved object: {e}")

    def _setup_template(self, img, name="Monitored Object"):
        """Internal helper to configure template image and compute features."""
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 320:
            scale = 320.0 / max_dim
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        self.template_bgr = img.copy()
        self.template_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.template_h, self.template_w = self.template_gray.shape[:2]
        self.object_name = name or "Monitored Object"

        # Precompute SIFT features
        if self.has_sift and self.sift is not None:
            try:
                self.template_sift_kp, self.template_sift_des = self.sift.detectAndCompute(self.template_gray, None)
            except Exception:
                self.template_sift_kp, self.template_sift_des = None, None

        # Precompute ORB features
        try:
            self.template_orb_kp, self.template_orb_des = self.orb.detectAndCompute(self.template_gray, None)
        except Exception:
            self.template_orb_kp, self.template_orb_des = None, None

        self.has_object = True
        # Start unverified until camera detects it
        self.is_present = False
        self.consecutive_misses = 0
        self.consecutive_hits = 0
        self.last_seen_time = None
        self.missing_start_time = time.time()
        start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self.active_interval_id = start_missing_interval(self.object_name, start_str)
        except Exception:
            self.active_interval_id = None
        self.alert_newly_triggered = True
        self.restored_newly_triggered = False

    def set_object_image(self, img, name="Monitored Object"):
        """
        Register a new object image to monitor.
        img: BGR numpy ndarray
        """
        if img is None or img.size == 0:
            return False

        with self.lock:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            cv2.imwrite(self.storage_path, img)
            self._setup_template(img, name)

        return True

    def clear_object(self):
        """Remove the current monitored object."""
        with self.lock:
            if self.active_interval_id:
                try:
                    end_missing_interval(
                        self.active_interval_id,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        0,
                        "Monitoring cleared"
                    )
                except Exception:
                    pass

            self.has_object = False
            self.template_bgr = None
            self.template_gray = None
            self.template_sift_kp = None
            self.template_sift_des = None
            self.template_orb_kp = None
            self.template_orb_des = None
            self.is_present = False
            self.object_name = ""
            self.consecutive_misses = 0
            self.consecutive_hits = 0
            self.last_seen_time = None
            self.missing_start_time = None
            self.active_interval_id = None
            self.alert_newly_triggered = False
            self.restored_newly_triggered = False

            if os.path.exists(self.storage_path):
                try:
                    os.remove(self.storage_path)
                except Exception:
                    pass

    def update_frame(self, frame):
        """
        Feed newest camera frame to background monitor.
        Does not block camera thread.
        """
        if not self.has_object or frame is None:
            return

        with self.lock:
            self._latest_frame = frame.copy()
            self._frame_version += 1

    def _worker_loop(self):
        """Background thread executing matching without delaying camera capture."""
        while not self._stop_event.is_set():
            frame_to_check = None
            with self.lock:
                if self.has_object and self._frame_version != self._processed_version:
                    frame_to_check = self._latest_frame
                    self._processed_version = self._frame_version

            if frame_to_check is None:
                self._stop_event.wait(0.04)
                continue

            self._match_frame(frame_to_check)
            self._stop_event.wait(0.04)

    def _match_frame(self, frame):
        """
        Evaluate whether template object exists in current camera frame.
        Uses SIFT + RANSAC homography, multi-scale template matching, and ORB.
        """
        if not self.has_object or self.template_gray is None:
            return

        h, w = frame.shape[:2]
        if w > 540:
            f_scale = 540.0 / w
            frame_small = cv2.resize(frame, (int(w * f_scale), int(h * f_scale)), interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame

        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        fh, fw = frame_gray.shape[:2]

        matched = False
        sift_inliers = 0
        good_sift_matches = 0
        best_tm_score = 0.0
        good_orb_matches = 0

        # 1. SIFT + FLANN + RANSAC HOMOGRAPHY
        if self.has_sift and self.sift is not None and self.template_sift_des is not None and len(self.template_sift_des) >= 4:
            try:
                kp_frame, des_frame = self.sift.detectAndCompute(frame_gray, None)
                if des_frame is not None and len(des_frame) >= 4:
                    raw_matches = self.flann.knnMatch(self.template_sift_des, des_frame, k=2)
                    good_sift = []
                    for pair in raw_matches:
                        if len(pair) == 2:
                            m, n = pair
                            if m.distance < 0.72 * n.distance:
                                good_sift.append(m)
                    good_sift_matches = len(good_sift)

                    if good_sift_matches >= 4:
                        src_pts = np.float32([self.template_sift_kp[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)
                        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                        if mask is not None:
                            sift_inliers = int(np.sum(mask))
            except Exception:
                pass

        # 2. MULTI-SCALE TEMPLATE MATCHING
        scales = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
        for s in scales:
            tw = int(self.template_w * s)
            th = int(self.template_h * s)
            if 30 <= tw < fw and 30 <= th < fh:
                try:
                    t_scaled = cv2.resize(self.template_gray, (tw, th))
                    res = cv2.matchTemplate(frame_gray, t_scaled, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_tm_score:
                        best_tm_score = float(max_val)
                except Exception:
                    pass

        # 3. ORB COMPLEMENTARY MATCHING
        if self.template_orb_des is not None and len(self.template_orb_des) >= 8:
            try:
                kp_orb_f, des_orb_f = self.orb.detectAndCompute(frame_gray, None)
                if des_orb_f is not None and len(des_orb_f) >= 8:
                    raw_orb = self.matcher.knnMatch(self.template_orb_des, des_orb_f, k=2)
                    for pair in raw_orb:
                        if len(pair) == 2:
                            m, n = pair
                            if m.distance < 0.75 * n.distance:
                                good_orb_matches += 1
            except Exception:
                pass

        # DETECTION DECISION - Robust multi-modal verification
        # Avoids false positives on background walls; requires genuine feature support
        matched = bool(
            sift_inliers >= 5
            or (good_sift_matches >= 8 and sift_inliers >= 4)
            or (best_tm_score >= 0.76 and (sift_inliers >= 2 or good_sift_matches >= 3 or good_orb_matches >= 4))
            or (best_tm_score >= 0.82)
            or (good_orb_matches >= 12 and best_tm_score >= 0.62)
        )

        now = time.time()
        with self.lock:
            computed_conf = max(
                best_tm_score,
                min(sift_inliers / 10.0, 1.0),
                min(good_sift_matches / 15.0, 1.0),
                min(good_orb_matches / 15.0, 1.0)
            )
            self.confidence = float(computed_conf)

            if matched:
                self.consecutive_hits += 1
                self.consecutive_misses = 0
                self.last_seen_time = now

                # Object restored from missing state
                if not self.is_present:
                    self.is_present = True
                    self.restored_newly_triggered = True

                    if self.missing_start_time is not None and self.active_interval_id is not None:
                        duration = max(0.0, now - self.missing_start_time)
                        mins = int(duration // 60)
                        secs = int(duration % 60)
                        dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                        end_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

                        try:
                            end_missing_interval(
                                self.active_interval_id,
                                end_str,
                                duration,
                                dur_str
                            )
                        except Exception as db_e:
                            print(f"[ObjectMonitor] DB interval end error: {db_e}")

                        self.active_interval_id = None
                        self.missing_start_time = None

            else:
                self.consecutive_hits = 0
                self.consecutive_misses += 1

                time_since_seen = (now - self.last_seen_time) if self.last_seen_time else 999.0

                if time_since_seen > self.GRACE_PERIOD_SECONDS and self.consecutive_misses >= self.MIN_MISSES_FOR_ALERT:
                    if self.is_present:
                        self.is_present = False
                        self.missing_start_time = self.last_seen_time or now
                        self.alert_newly_triggered = True
                        start_str = datetime.fromtimestamp(self.missing_start_time).strftime("%Y-%m-%d %H:%M:%S")

                        try:
                            self.active_interval_id = start_missing_interval(
                                self.object_name,
                                start_str
                            )
                        except Exception as db_e:
                            print(f"[ObjectMonitor] DB interval start error: {db_e}")

    def check_and_reset_alert_trigger(self):
        """Returns True once when object transitions to missing."""
        with self.lock:
            if self.alert_newly_triggered:
                self.alert_newly_triggered = False
                return True
            return False

    def check_and_reset_restored_trigger(self):
        """Returns True once when object transitions back to present."""
        with self.lock:
            if self.restored_newly_triggered:
                self.restored_newly_triggered = False
                return True
            return False

    def get_status(self):
        """Return clean status dictionary for dashboard, reports, and query assistant."""
        with self.lock:
            if not self.has_object:
                return {
                    "has_object": False,
                    "object_name": "",
                    "is_present": False,
                    "alert": False,
                    "confidence": 0.0,
                    "status_text": "NOT CONFIGURED",
                    "badge": "NOT SET",
                    "badge_color": "#94a3b8",
                    "last_seen": None,
                    "last_seen_formatted": "Never",
                    "missing_start": None,
                    "missing_start_formatted": None,
                    "missing_duration_seconds": 0.0
                }

            is_present = self.is_present
            alert = not is_present
            now = time.time()

            last_seen_fmt = (
                datetime.fromtimestamp(self.last_seen_time).strftime("%Y-%m-%d %I:%M:%S %p")
                if self.last_seen_time
                else "Never recorded"
            )

            missing_start_fmt = (
                datetime.fromtimestamp(self.missing_start_time).strftime("%Y-%m-%d %I:%M:%S %p")
                if self.missing_start_time
                else None
            )

            missing_duration = (now - self.missing_start_time) if (not is_present and self.missing_start_time) else 0.0

            return {
                "has_object": True,
                "object_name": self.object_name,
                "is_present": is_present,
                "alert": alert,
                "confidence": round(self.confidence, 2),
                "status_text": "SAFE / IN VIEW" if is_present else "ALERT: NOT DETECTED",
                "badge": "IN VIEW" if is_present else "NOT DETECTED (ALERT)",
                "badge_color": "#36d17c" if is_present else "#ef4444",
                "last_seen": self.last_seen_time,
                "last_seen_formatted": last_seen_fmt,
                "missing_start": self.missing_start_time,
                "missing_start_formatted": missing_start_fmt,
                "missing_duration_seconds": round(missing_duration, 1)
            }

    def stop(self):
        """Stop background worker on shutdown."""
        self._stop_event.set()


# Global singleton instance
object_monitor = ObjectMonitor()
