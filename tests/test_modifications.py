"""
Unit and Integration Tests for SentriX Examiner Modifications
Covers:
1. Answering queries based on existing data (Query Assistant Engine)
2. Visualizations in the report (Matplotlib analytics embedded in PDF reports)
3. Tamper detection followed by siren trigger (Lens covered, blur, camera moved)
4. Monitored Object Watcher (Upload object image, robust presence check, alert & report timeline)
5. Video recordings: collapsible down button, date-wise grouping, minimum 1-minute duration
"""

import unittest
import numpy as np
import cv2
import io
import os
import time

from app.query.engine import query_engine
from app.reports.visualizer import report_visualizer
from app.reports.generator import generate_daily_report
from app.tamper.siren import SirenManager, siren_manager
from app.object_monitor import ObjectMonitor, object_monitor
from app.database.db import (
    get_events_by_date,
    get_event_type_distribution,
    get_hourly_activity,
    get_tamper_summary,
    get_recordings_summary
)


class TestSentrixExaminerModifications(unittest.TestCase):

    # --------------------------------------------------------
    # 1. QUERY ENGINE TESTS (EXAMINER RECOMMENDATION 1)
    # --------------------------------------------------------
    def test_query_who_detected(self):
        res = query_engine.query("Who was detected today?")
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "who_detected")
        self.assertIn("personnel", res["answer"].lower())

    def test_query_tamper(self):
        res = query_engine.query("Was there any camera tampering?")
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "tamper_incidents")

    def test_query_recordings(self):
        res = query_engine.query("How many recordings were made?")
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "recordings")

    def test_query_summary(self):
        res = query_engine.query("Security summary for today")
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "summary")

    def test_query_person_specific(self):
        res = query_engine.query("When was Dharshana seen?")
        self.assertTrue(res["success"])
        self.assertIn("dharshana", res["answer"].lower())

    def test_query_tamper_siren(self):
        res = query_engine.query("Was deterrent siren triggered during camera tampering?")
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "tamper_incidents")
        self.assertIn("siren", res["answer"].lower())

    def test_query_object_last_there(self):
        res = query_engine.query("When was the object last there?")
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "monitored_object")
        self.assertTrue(any(w in res["answer"].lower() for w in ["object", "monitored", "asset"]))

    def test_query_object_missing(self):
        res = query_engine.query("When was the object missing?")
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "monitored_object")
        self.assertTrue(any(w in res["answer"].lower() for w in ["object", "missing", "asset"]))

    # --------------------------------------------------------
    # 2. REPORT VISUALIZATION TESTS (EXAMINER RECOMMENDATION 2)
    # --------------------------------------------------------
    def test_visualizer_donut_chart(self):
        dist = {"motion": 10, "face_recognized": 5, "tamper": 2}
        buf = report_visualizer.generate_event_distribution_chart(dist)
        self.assertIsNotNone(buf)
        self.assertGreater(len(buf.getvalue()), 1000)

    def test_visualizer_timeline_chart(self):
        hourly = [0] * 24
        hourly[10] = 5
        hourly[11] = 12
        buf = report_visualizer.generate_hourly_timeline_chart(hourly)
        self.assertIsNotNone(buf)
        self.assertGreater(len(buf.getvalue()), 1000)

    def test_visualizer_incidents_chart(self):
        incidents = {"CAMERA MOVED": 3, "LENS COVERED": 1, "SIREN ALARM": 2}
        buf = report_visualizer.generate_security_incidents_chart(incidents)
        self.assertIsNotNone(buf)
        self.assertGreater(len(buf.getvalue()), 1000)

    def test_daily_report_generation_with_charts(self):
        dummy_events = [
            "2026-10-31 10:15:00 - Motion detected",
            "2026-10-31 10:20:00 - Monitored object not detected in camera view! (Missing Alert)"
        ]
        pdf_path = generate_daily_report("2026-10-31", dummy_events, [])
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 10000)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        self.assertNotIn(b"Security Event Timeline", pdf_bytes)
        self.assertIn(b"/Subtype /Image", pdf_bytes)

    # --------------------------------------------------------
    # 3. TAMPER SIREN DETERRENT TESTS (EXAMINER RECOMMENDATION 4)
    # --------------------------------------------------------
    def test_siren_lifecycle(self):
        siren = SirenManager(default_duration=2.0, sound_enabled=False)
        self.assertFalse(siren.is_active())

        # Trigger siren
        siren.trigger(reason="TEST CAMERA TAMPER", duration=2.0)
        self.assertTrue(siren.is_active())
        status = siren.get_status()
        self.assertTrue(status["active"])
        self.assertEqual(status["reason"], "TEST CAMERA TAMPER")

        # Mute siren
        siren.mute()
        self.assertTrue(siren.get_status()["muted"])

        # Stop siren
        siren.stop()
        self.assertFalse(siren.is_active())
        self.assertFalse(siren.get_status()["active"])

    # --------------------------------------------------------
    # 4. DATABASE ANALYTICS HELPERS
    # --------------------------------------------------------
    def test_database_analytics(self):
        dist = get_event_type_distribution()
        self.assertIsInstance(dist, dict)
        self.assertIn("motion", dist)

        hourly = get_hourly_activity("2026-10-31")
        self.assertEqual(len(hourly), 24)

        tamper_summary = get_tamper_summary()
        self.assertIsInstance(tamper_summary, dict)

        rec_summary = get_recordings_summary()
        self.assertIn("total_count", rec_summary)

    # --------------------------------------------------------
    # 5. DATE-WISE RECORDINGS AND MINIMUM DURATION TESTS
    # --------------------------------------------------------
    def test_recordings_api_datewise(self):
        from app.dashboard.dashboard import app
        client = app.test_client()
        resp = client.get("/api/recordings")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("by_date", data)
        self.assertIn("dates", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["by_date"], dict)

    def test_min_recording_duration_constant(self):
        import app.main as main_mod
        self.assertTrue(hasattr(main_mod, "MIN_RECORDING_SECONDS"))
        self.assertEqual(main_mod.MIN_RECORDING_SECONDS, 60.0)

    # --------------------------------------------------------
    # 6. MONITORED OBJECT WATCHER TESTS
    # --------------------------------------------------------
    def test_object_monitor_lifecycle(self):
        test_monitor = ObjectMonitor(storage_path="data/test_monitored_obj.png")
        try:
            self.assertFalse(test_monitor.get_status()["has_object"])

            # 1. Set object
            obj_img = np.ones((80, 80, 3), dtype=np.uint8) * 180
            cv2.circle(obj_img, (40, 40), 20, (0, 0, 255), -1)
            ok = test_monitor.set_object_image(obj_img, name="Red Badge")
            self.assertTrue(ok)
            self.assertTrue(test_monitor.get_status()["has_object"])
            self.assertEqual(test_monitor.get_status()["object_name"], "Red Badge")

            # 2. Clear object
            test_monitor.clear_object()
            self.assertFalse(test_monitor.get_status()["has_object"])
        finally:
            test_monitor.stop()
            for p in ['data/test_monitored_obj.png', 'data/test_detect_obj.png']:
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass

    def test_object_monitor_detection_and_alert(self):
        test_monitor = ObjectMonitor(storage_path="data/test_detect_obj.png")
        try:
            test_monitor.GRACE_PERIOD_SECONDS = 0.05
            test_monitor.MIN_MISSES_FOR_ALERT = 3

            # Create distinct pattern object
            obj_img = np.zeros((80, 80, 3), dtype=np.uint8)
            cv2.rectangle(obj_img, (10, 10), (70, 70), (255, 255, 255), -1)
            cv2.circle(obj_img, (40, 40), 15, (0, 128, 255), -1)
            test_monitor.set_object_image(obj_img, name="Target Card")

            # Scene WITH object
            scene_with_obj = np.zeros((300, 400, 3), dtype=np.uint8)
            scene_with_obj[50:130, 80:160] = obj_img

            # Test matching on scene containing object
            test_monitor._match_frame(scene_with_obj)
            status = test_monitor.get_status()
            self.assertTrue(status["is_present"])
            self.assertFalse(status["alert"])

            # Scene WITHOUT object (blank / missing)
            time.sleep(0.06)
            blank_scene = np.zeros((300, 400, 3), dtype=np.uint8)
            for _ in range(6):
                test_monitor._match_frame(blank_scene)

            # Object is now flagged as missing / not detected
            missing_status = test_monitor.get_status()
            self.assertFalse(missing_status["is_present"])
            self.assertTrue(missing_status["alert"])
            self.assertIn("ALERT", missing_status["status_text"])
        finally:
            test_monitor.stop()
            for p in ['data/test_monitored_obj.png', 'data/test_detect_obj.png']:
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass

    def test_object_monitor_api_endpoints(self):
        from app.dashboard.dashboard import app
        from app.object_monitor import object_monitor
        client = app.test_client()

        existing_path = "data/monitored_object.png"
        backup_bytes = None
        backup_name = None
        if os.path.exists(existing_path):
            try:
                with open(existing_path, "rb") as f:
                    backup_bytes = f.read()
                backup_name = getattr(object_monitor, "object_name", "Monitored Object")
            except Exception:
                pass

        try:
            # Check status
            resp = client.get("/api/object-monitor/status")
            self.assertEqual(resp.status_code, 200)

            # Upload image
            test_img = np.ones((80, 80, 3), dtype=np.uint8) * 150
            _, png_bytes = cv2.imencode(".png", test_img)
            upload_data = {"file": (io.BytesIO(png_bytes.tobytes()), "office_keys.png")}
            up_resp = client.post("/api/object-monitor/upload", data=upload_data, content_type="multipart/form-data")
            self.assertEqual(up_resp.status_code, 200)
            self.assertTrue(up_resp.get_json()["success"])

            # Fetch image
            img_resp = client.get("/api/object-monitor/image")
            self.assertEqual(img_resp.status_code, 200)

            # Clear
            clear_resp = client.post("/api/object-monitor/clear")
            self.assertEqual(clear_resp.status_code, 200)
            self.assertFalse(clear_resp.get_json()["status"]["has_object"])
        finally:
            if backup_bytes is not None:
                np_arr = np.frombuffer(backup_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    object_monitor.set_object_image(img, name=backup_name)

    # --------------------------------------------------------
    # 7. MISSING TIME INTERVAL TRACKING & SINGLE-ALERT TRIGGER
    # --------------------------------------------------------
    def test_missing_intervals_database(self):
        from app.database.db import (
            start_missing_interval,
            end_missing_interval,
            get_missing_intervals
        )
        # Start interval
        int_id = start_missing_interval("Test Vault Asset", "2026-11-01 10:00:00")
        self.assertIsNotNone(int_id)

        # Query active interval
        intervals = get_missing_intervals("2026-11-01")
        self.assertTrue(any(i["id"] == int_id for i in intervals))
        active = next(i for i in intervals if i["id"] == int_id)
        self.assertEqual(active["status"], "CURRENTLY MISSING")

        # End interval
        end_missing_interval(int_id, "2026-11-01 10:05:30", 330, "5 mins 30 secs")
        intervals_after = get_missing_intervals("2026-11-01")
        ended = next(i for i in intervals_after if i["id"] == int_id)
        self.assertEqual(ended["status"], "RESTORED")
        self.assertEqual(ended["duration_str"], "5 mins 30 secs")

    def test_single_alert_trigger_edge(self):
        monitor = ObjectMonitor(storage_path="data/test_trigger_obj.png")
        try:
            monitor.GRACE_PERIOD_SECONDS = 0.05
            monitor.MIN_MISSES_FOR_ALERT = 2

            obj_img = np.zeros((80, 80, 3), dtype=np.uint8)
            cv2.rectangle(obj_img, (10, 10), (70, 70), (255, 255, 255), -1)
            monitor.set_object_image(obj_img, name="Trigger Test Key")

            # Presence check
            scene_with_obj = np.zeros((300, 400, 3), dtype=np.uint8)
            scene_with_obj[50:130, 80:160] = obj_img
            monitor._match_frame(scene_with_obj)
            self.assertTrue(monitor.get_status()["is_present"])

            # Disappear
            time.sleep(0.06)
            blank_scene = np.zeros((300, 400, 3), dtype=np.uint8)
            monitor._match_frame(blank_scene)
            monitor._match_frame(blank_scene)
            monitor._match_frame(blank_scene)

            # Trigger should be True EXACTLY ONCE
            self.assertTrue(monitor.check_and_reset_alert_trigger())
            # Second check immediately False (no repeated flooding)
            self.assertFalse(monitor.check_and_reset_alert_trigger())
        finally:
            monitor.stop()
            if os.path.exists('data/test_trigger_obj.png'):
                try: os.remove('data/test_trigger_obj.png')
                except Exception: pass

    def test_report_generation_with_missing_intervals(self):
        from app.database.db import start_missing_interval, end_missing_interval
        from app.reports.generator import generate_daily_report

        i_id = start_missing_interval("High Value Hardware", "2026-11-01 12:00:00")
        end_missing_interval(i_id, "2026-11-01 12:14:00", 840, "14 mins 0 secs")

        pdf_path = generate_daily_report("2026-11-01", [], [])
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)

    def test_query_missing_interval_time_period(self):
        from app.database.db import start_missing_interval, end_missing_interval
        from app.query.engine import query_engine

        int_id = start_missing_interval("Office Projector", "2026-11-01 09:10:00")
        end_missing_interval(int_id, "2026-11-01 09:45:00", 2100, "35 mins 0 secs")

        res = query_engine.query("When was the object missing?")
        self.assertTrue(res["success"])
        self.assertIn("Office Projector", res["answer"])
        self.assertIn("09:10:00", res["answer"])

    # --------------------------------------------------------
    # 8. BLUR ROBUSTNESS & LENS-COVERED-ONLY SIREN TESTS
    # --------------------------------------------------------
    def test_blur_detector_robustness(self):
        from app.tamper.blur import BlurDetector
        detector = BlurDetector(threshold=6.0, confirm_frames=4, calibration_frames=10, ratio=0.20)
        
        # 1. Normal indoor scene (even low texture) should NOT trigger blur
        plain_frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
        noise = np.random.normal(0, 2, (480, 640, 3)).astype(np.int16)
        plain_frame = np.clip(plain_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        for _ in range(15):
            detector.detect(plain_frame)

        blurry = False
        for _ in range(10):
            b, _ = detector.detect(plain_frame)
            if b:
                blurry = True
        self.assertFalse(blurry, "Normal camera scene was falsely detected as camera blurred!")

        # 2. Heavily blurred scene (tampering) MUST be detected
        tamper_blurred = cv2.GaussianBlur(plain_frame, (31, 31), 10)
        detected_blur = False
        for _ in range(8):
            b, _ = detector.detect(tamper_blurred)
            if b:
                detected_blur = True
        self.assertTrue(detected_blur, "Tamper blur was not detected!")

    def test_tamper_siren_lens_covered_only(self):
        from app.tamper.detector import TamperDetector
        from app.tamper.siren import SirenManager

        siren = SirenManager(default_duration=2.0, sound_enabled=False)
        ref_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(ref_frame, 'TEST SCENE', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.rectangle(ref_frame, (100, 200), (500, 400), (200, 200, 200), -1)
        td = TamperDetector(ref_frame)

        # Calibrate with normal frame
        for _ in range(15):
            res = td.detect(ref_frame)
        self.assertEqual(res["status"], "CAMERA OK")
        if res["status"] == "LENS COVERED":
            siren.trigger(reason="TAMPER: LENS COVERED")
        self.assertFalse(siren.is_active())

        # Blur frame -> CAMERA BLURRED -> Siren MUST NOT trigger
        blurred = cv2.GaussianBlur(ref_frame, (31, 31), 10)
        for _ in range(10):
            res_b = td.detect(blurred)
        self.assertEqual(res_b["status"], "CAMERA BLURRED")
        if res_b["status"] == "LENS COVERED":
            siren.trigger(reason="TAMPER: LENS COVERED")
        self.assertFalse(siren.is_active(), "Siren triggered on blur! Siren must be for lens covered only.")

        # Dark frame -> LENS COVERED -> Siren triggers
        dark = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            res_dark = td.detect(dark)
        self.assertEqual(res_dark["status"], "LENS COVERED")
        if res_dark["status"] == "LENS COVERED":
            siren.trigger(reason="TAMPER: LENS COVERED")
        self.assertTrue(siren.is_active(), "Siren did not trigger on lens covered!")
        siren.stop()


if __name__ == "__main__":
    unittest.main()
