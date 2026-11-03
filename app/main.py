import cv2
import os
import time
import threading
from datetime import datetime

from mtcnn import MTCNN
from keras_facenet import FaceNet

from app.motion.detector import MotionDetector
from app.face.recognizer import load_registered_faces, recognize_face
from app.tamper.detector import TamperDetector
from app.tamper.siren import siren_manager
from app.object_monitor import object_monitor

from app.database.db import (
    add_event as add_database_event,
    add_recording,
    add_tamper_event
)

from app.dashboard.dashboard import (
    start_dashboard,
    update_frame,
    update_status,
    add_event
)


# ============================================================
# SENTRIX CONFIGURATION
# ============================================================

POST_MOTION_SECONDS = 10
MIN_RECORDING_SECONDS = 60.0  # Minimum 1 minute (60 seconds) recording duration
EVENT_COOLDOWN = 5  # Default cooldown; recognition/tamper events use longer cooldowns
OUTPUT_FOLDER = "recordings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# EVENT CONTROL
# ============================================================

last_event_times = {}


# ============================================================
# EVENT LOGGER
# ============================================================

def log_event(
    message,
    event_key=None,
    person_name=None,
    confidence=None,
    tamper_type=None,
    brightness=None,
    blur_variance=None,
    histogram_correlation=None,
    event_type=None,
    cooldown=None,
    **kwargs
):
    """
    Logs events to:
    1. Console
    2. Dashboard
    3. SQLite database
    """

    global last_event_times

    if event_key is None:
        event_key = message

    current_time = time.time()

    # Use specified cooldown if provided, else use defaults
    if cooldown is not None:
        event_cooldown = float(cooldown)
    elif event_key.startswith("recognized_"):
        event_cooldown = 10
    elif event_key == "unknown_person":
        event_cooldown = 8
    elif event_key.startswith("tamper_"):
        event_cooldown = 3
    elif event_key == "obj_missing_alert":
        event_cooldown = 15
    else:
        event_cooldown = EVENT_COOLDOWN

    if event_key in last_event_times:
        elapsed = current_time - last_event_times[event_key]
        if elapsed < event_cooldown:
            return

    last_event_times[event_key] = current_time

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    full_message = (
        f"{timestamp} - {message}"
    )

    # --------------------------------------------------------
    # CONSOLE
    # --------------------------------------------------------

    print(full_message)

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    try:

        add_event(full_message)

    except Exception as e:

        print(
            "Dashboard event error:",
            e
        )

    # --------------------------------------------------------
    # DATABASE EVENT TYPE
    # --------------------------------------------------------

    try:

        if event_type is None:
            message_lower = message.lower()

            if "motion detected" in message_lower:
                event_type = "motion"
            elif "siren" in message_lower:
                event_type = "siren_trigger"
            elif "tamper detected" in message_lower:
                event_type = "tamper"
            elif "unknown person" in message_lower:
                event_type = "unknown_person"
            elif "recognized" in message_lower:
                event_type = "face_recognized"
            elif "recording started" in message_lower:
                event_type = "recording_start"
            elif "recording stopped" in message_lower:
                event_type = "recording_stop"
            elif "monitored object" in message_lower or "missing alert" in message_lower:
                event_type = "object_alert"
            else:
                event_type = "system"

        add_database_event(
            event_type=event_type,
            message=message,
            person_name=person_name,
            confidence=confidence
        )

    except Exception as e:

        print(
            "Database event error:",
            e
        )

    # --------------------------------------------------------
    # TAMPER DATABASE
    # --------------------------------------------------------

    if (
        tamper_type is not None
        and "tamper detected" in message.lower()
    ):

        try:

            add_tamper_event(
                tamper_type=tamper_type,
                brightness=brightness,
                blur_variance=blur_variance,
                histogram_correlation=(
                    histogram_correlation
                )
            )

        except Exception as e:

            print(
                "Tamper database error:",
                e
            )


# ============================================================
# DASHBOARD STATUS
# ============================================================

def send_dashboard_status(
    camera="ACTIVE",
    motion=False,
    face_detected=False,
    people=None,
    tamper="CAMERA OK",
    tampering=False,
    recording=False,
    siren=False,
    object_monitor_status=None
):

    if people is None:
        people = []

    status = {

        "camera": camera,

        "motion": bool(
            motion
        ),

        "face_detected": bool(
            face_detected
        ),

        "people": list(
            people
        ),

        "tamper": str(
            tamper
        ),

        "tampering": bool(
            tampering
        ),

        "recording": bool(
            recording
        ),

        "siren": bool(
            siren
        ),

        "object_monitor": (
            object_monitor_status
            if object_monitor_status is not None
            else object_monitor.get_status()
        )
    }

    try:

        update_status(
            status
        )

    except Exception as e:

        print(
            "Dashboard status update error:",
            e
        )


# ============================================================
# MAIN SENTRIX ENGINE
# ============================================================

def finalize_mp4(temp_path, final_path):
    """Validate a closed MP4 and publish it under its final filename."""
    try:
        if not temp_path or not os.path.isfile(temp_path) or os.path.getsize(temp_path) < 1024:
            return False
        check = cv2.VideoCapture(temp_path)
        opened = check.isOpened()
        ok, first_frame = check.read() if opened else (False, None)
        check.release()
        if not opened or not ok or first_frame is None:
            return False
        os.replace(temp_path, final_path)
        return os.path.isfile(final_path) and os.path.getsize(final_path) >= 1024
    except Exception as exc:
        print("Video finalization error:", exc)
        return False


def run_sentrix():

    print()
    print("==============================")
    print("       SENTRIX STARTING")
    print("==============================")
    print()

    # ========================================================
    # CAMERA
    # ========================================================

    print(
        "Opening camera..."
    )

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_MSMF
    )

    if not camera.isOpened():

        print(
            "MSMF camera backend failed."
        )

        print(
            "Trying default camera backend..."
        )

        camera.release()

        camera = cv2.VideoCapture(
            0
        )

    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open camera."
        )
        print()

        send_dashboard_status(
            camera="ERROR"
        )

        return

    print(
        "Camera opened successfully."
    )

    # Reduce camera latency and avoid processing a backlog of old frames.
    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    print()

    # ========================================================
    # MOTION DETECTOR
    # ========================================================

    print(
        "Initializing motion detector..."
    )

    try:

        motion_detector = MotionDetector()

        print(
            "Motion detector ready."
        )

    except Exception as e:

        print(
            "ERROR initializing motion detector:"
        )

        print(e)

        camera.release()

        send_dashboard_status(
            camera="ERROR"
        )

        return

    print()

    # ========================================================
    # FACE DETECTOR
    # ========================================================

    print(
        "Loading MTCNN face detector..."
    )

    try:

        face_detector = MTCNN()

        print(
            "Face detector ready."
        )

    except Exception as e:

        print(
            "ERROR initializing face detector:"
        )

        print(e)

        camera.release()

        send_dashboard_status(
            camera="ERROR"
        )

        return

    print()

    # ========================================================
    # FACENET
    # ========================================================

    print(
        "Loading FaceNet model..."
    )

    try:

        embedder = FaceNet()

        print(
            "FaceNet model loaded successfully."
        )

    except Exception as e:

        print(
            "ERROR loading FaceNet:"
        )

        print(e)

        camera.release()

        send_dashboard_status(
            camera="ERROR"
        )

        return

    print()

    # ========================================================
    # REGISTERED FACES
    # ========================================================

    print(
        "Loading registered faces..."
    )

    try:

        registered_faces = (
            load_registered_faces()
        )
        last_registered_faces_check = time.time()

    except Exception as e:

        print(
            "ERROR loading registered faces:"
        )

        print(e)

        registered_faces = {}

    print()

    if registered_faces:

        print(
            "Registered faces:"
        )

        for person_name, embeddings in (
            registered_faces.items()
        ):

            try:

                embedding_count = len(
                    embeddings
                )

            except Exception:

                embedding_count = 1

            print(
                f"  {person_name}: "
                f"{embedding_count} embeddings"
            )

    else:

        print(
            "WARNING: No registered faces found."
        )

    print()

    # ========================================================
    # TAMPER DETECTOR
    # ========================================================

    tamper_detector = None

    # ========================================================
    # RECORDING VARIABLES
    # ========================================================

    recording = False

    video_writer = None

    last_motion_time = 0

    current_recording_path = None

    current_recording_temp_path = None

    recording_start_time = None

    # ========================================================
    # CURRENT TAMPER STATUS
    # ========================================================

    current_tamper_status = "CAMERA OK"
    # Tracks status transitions so the same tamper alert is not logged every few seconds.
    last_logged_tamper_status = "CAMERA OK"

    # Run expensive tamper analysis at most five times per second.
    # Reuse the latest result between checks to reduce processing overhead.
    TAMPER_CHECK_INTERVAL = 0.08
    last_tamper_check_time = 0.0
    cached_tamper_result = {
        "tampering": False,
        "status": "CAMERA OK",
        "brightness": 0.0,
        "blur_variance": 0.0,
        "histogram_correlation": 1.0
    }

    # Face inference runs in a worker thread so MTCNN/FaceNet do not block
    # camera capture, recording, or dashboard refresh.
    face_state_lock = threading.Lock()
    face_input = {"frame": None, "version": 0}
    face_output = {"version": -1, "faces": []}
    face_worker_stop = threading.Event()

    def face_inference_worker():
        last_version = -1
        while not face_worker_stop.is_set():
            with face_state_lock:
                version = face_input["version"]
                sample = face_input["frame"]
                known_faces = registered_faces

            if sample is None or version == last_version:
                face_worker_stop.wait(0.015)
                continue

            last_version = version
            try:
                # MTCNN and FaceNet both use RGB input. Keep the same colour
                # format for detection, cropping, registration, and recognition.
                sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                detections = face_detector.detect_faces(sample_rgb) or []
                faces = []
                sample_h, sample_w = sample_rgb.shape[:2]

                for detection in detections:
                    if float(detection.get("confidence", 0.0)) < 0.90:
                        continue
                    x, y, w, h = map(int, detection.get("box", (0, 0, 0, 0)))
                    x = max(0, x); y = max(0, y)
                    x2 = min(sample_w, x + w); y2 = min(sample_h, y + h)
                    if x2 <= x or y2 <= y:
                        continue
                    # IMPORTANT: crop from RGB frame, not the original BGR frame.
                    face_crop = sample_rgb[y:y2, x:x2]
                    if face_crop.size == 0:
                        continue
                    try:
                        name, score = recognize_face(face_crop, embedder, known_faces)
                    except Exception as exc:
                        print("Face recognition error:", exc)
                        name, score = "ERROR", 0.0

                    if name == "UNKNOWN":
                        display_name, label, color = "Unknown Person", f"Unknown ({score:.2f})", (0, 0, 255)
                        log_event(f"Unknown person detected ({score:.2f})", "unknown_person", confidence=score)
                    elif name == "ERROR":
                        display_name, label, color = "Recognition Error", "Recognition Error", (0, 255, 255)
                    else:
                        display_name, label, color = name, f"{name} ({score:.2f})", (0, 255, 0)
                        log_event(f"{name} recognized ({score:.2f})", f"recognized_{name}", person_name=name, confidence=score)

                    faces.append((x, y, x2, y2, label, color, display_name))

                with face_state_lock:
                    face_output["version"] = version
                    face_output["faces"] = faces
            except Exception as exc:
                print("Face detection worker error:", exc)
                with face_state_lock:
                    face_output["version"] = version
                    face_output["faces"] = []

            # Keep the detector responsive without repeatedly processing the same frame.
            face_worker_stop.wait(0.12)

    face_worker_thread = threading.Thread(
        target=face_inference_worker, name="SentriX-FaceWorker", daemon=True
    )
    face_worker_thread.start()
    last_dashboard_frame_update = 0.0
    DASHBOARD_FRAME_INTERVAL = 1.0 / 15.0  # Keep web dashboard smooth without overloading Flask.

    # ========================================================
    # INITIAL DASHBOARD STATUS
    # ========================================================

    send_dashboard_status(
        camera="ACTIVE",
        motion=False,
        face_detected=False,
        people=[],
        tamper="CAMERA OK",
        tampering=False,
        recording=False
    )

    # ========================================================
    # SENTRIX READY
    # ========================================================

    print("==============================")
    print("       SENTRIX RUNNING")
    print("==============================")
    print()

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    print(
        "Press Q in the camera window to quit."
    )

    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:
        if time.time() - last_registered_faces_check >= 5:
            try:
                registered_faces = load_registered_faces()
                last_registered_faces_check = time.time()
            except Exception as e:
                print("Face database reload error:", e)

        # ====================================================
        # CAPTURE FRAME
        # ====================================================

        ret, frame = camera.read()

        if not ret:

            print(
                "ERROR: Failed to read camera frame."
            )

            time.sleep(0.1)

            continue

        # ====================================================
        # FRAME SIZE
        # ====================================================

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ====================================================
        # RGB FRAME
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # INITIALIZE TAMPER DETECTOR
        # ====================================================

        if tamper_detector is None:

            try:

                tamper_detector = (
                    TamperDetector(frame)
                )

                print(
                    "Tamper detector initialized."
                )

            except Exception as e:

                print(
                    "Tamper detector initialization error:",
                    e
                )

                tamper_detector = None

        # ====================================================
        # MOTION DETECTION
        # ====================================================

        motion_detected = False

        try:

            motion_detected = (
                motion_detector.detect(
                    frame
                )
            )

        except Exception as e:

            print(
                "Motion detection error:",
                e
            )

        # ====================================================
        # MOTION EVENT
        # ====================================================

        if motion_detected:

            last_motion_time = time.time()

            log_event(
                "Motion detected",
                "motion"
            )

        # ====================================================
        # TAMPER DETECTION (THROTTLED)
        # ====================================================

        current_time = time.time()

        if (
            tamper_detector is not None
            and current_time - last_tamper_check_time >= TAMPER_CHECK_INTERVAL
        ):
            try:
                tamper_result = tamper_detector.detect(frame)
                last_tamper_check_time = current_time

                if isinstance(tamper_result, dict):
                    cached_tamper_result = {
                        "tampering": bool(tamper_result.get("tampering", False)),
                        "status": str(tamper_result.get("status", "CAMERA OK")),
                        "brightness": tamper_result.get("brightness"),
                        "blur_variance": tamper_result.get("blur_variance"),
                        "histogram_correlation": tamper_result.get("histogram_correlation")
                    }
                else:
                    cached_tamper_result = {
                        "tampering": str(tamper_result) != "CAMERA OK",
                        "status": str(tamper_result),
                        "brightness": None,
                        "blur_variance": None,
                        "histogram_correlation": None
                    }

            except Exception as e:
                print("Tamper detection error:", e)
                # Keep the last valid result if a check fails.

        tamper_detected = bool(cached_tamper_result.get("tampering", False))
        current_tamper_status = str(
            cached_tamper_result.get("status", "CAMERA OK")
        )
        tamper_brightness = cached_tamper_result.get("brightness")
        tamper_blur_variance = cached_tamper_result.get("blur_variance")
        tamper_histogram_correlation = cached_tamper_result.get(
            "histogram_correlation"
        )

        # Trigger siren ONLY for lens covered tampering (not for blur, moved, or unknown person)
        if current_tamper_status == "LENS COVERED":
            try:
                siren_manager.trigger(
                    reason="TAMPER: LENS COVERED",
                    duration=10.0
                )
            except Exception as siren_err:
                print("Siren trigger error:", siren_err)
        else:
            if siren_manager.is_active():
                siren_manager.stop()

        # Log only when the reported tamper status changes.
        if current_tamper_status != last_logged_tamper_status:
            if tamper_detected:
                siren_tag = " (Siren Triggered)" if current_tamper_status == "LENS COVERED" else ""
                log_event(
                    f"Tamper detected: {current_tamper_status}{siren_tag}",
                    event_key=f"tamper_{current_tamper_status}_{int(time.time())}",
                    cooldown=2.0,
                    tamper_type=current_tamper_status,
                    brightness=tamper_brightness,
                    blur_variance=tamper_blur_variance,
                    histogram_correlation=tamper_histogram_correlation
                )
            elif last_logged_tamper_status != "CAMERA OK":
                log_event(
                    "Camera tampering cleared",
                    event_key=f"tamper_cleared_{int(time.time())}",
                    cooldown=2.0
                )

            last_logged_tamper_status = current_tamper_status

        # ====================================================
        # ASYNCHRONOUS FACE DETECTION / DISPLAY
        # ====================================================

        # Publish the newest frame; the worker always skips stale frames.
        with face_state_lock:
            face_input["frame"] = frame.copy()
            face_input["version"] += 1
            faces_to_draw = list(face_output["faces"])

        face_detected = bool(faces_to_draw)
        detected_people = list(dict.fromkeys(face[6] for face in faces_to_draw))

        for x, y, x2, y2, label, box_color, _person_name in faces_to_draw:
            cv2.rectangle(frame, (x, y), (x2, y2), box_color, 2)
            cv2.putText(
                frame, label, (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2
            )

        # ====================================================
        # START RECORDING
        # ====================================================

        if (
            motion_detected
            and not recording
        ):

            timestamp = (
                datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            current_recording_path = os.path.join(OUTPUT_FOLDER, f"SENTRIX_{timestamp}.mp4")
            current_recording_temp_path = os.path.join(OUTPUT_FOLDER, f".SENTRIX_{timestamp}.recording.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = (
                cv2.VideoWriter(
                    current_recording_temp_path,
                    fourcc,
                    20.0,
                    (
                        frame_width,
                        frame_height
                    )
                )
            )

            if video_writer.isOpened():

                recording = True

                recording_start_time = (
                    time.time()
                )

                print()

                print(
                    "Recording started:",
                    current_recording_path
                )

                log_event(
                    (
                        f"Recording started: "
                        f"{current_recording_path}"
                    ),
                    "recording_start"
                )

            else:

                print(
                    "ERROR: Could not create video."
                )

                video_writer.release()

                video_writer = None

                current_recording_path = None
                current_recording_temp_path = None

        # ====================================================
        # WRITE VIDEO
        # ====================================================

        if (
            recording
            and video_writer is not None
        ):

            try:

                video_writer.write(
                    frame
                )

            except Exception as e:

                print(
                    "Video writing error:",
                    e
                )

        # ====================================================
        # STOP RECORDING
        # ====================================================

        if recording:

            elapsed_since_motion = (
                time.time()
                - last_motion_time
            )

            total_recording_elapsed = (
                time.time() - recording_start_time
            ) if recording_start_time is not None else 0

            # Only stop recording when:
            # 1. Total recording duration has reached at least 1 minute (60 seconds)
            # 2. Motion has been clear for at least POST_MOTION_SECONDS (10s buffer)
            if (
                total_recording_elapsed >= MIN_RECORDING_SECONDS
                and elapsed_since_motion >= POST_MOTION_SECONDS
            ):

                # --------------------------------------------
                # CALCULATE DURATION
                # --------------------------------------------

                recording_duration = 0

                if recording_start_time is not None:

                    recording_duration = (
                        time.time()
                        - recording_start_time
                    )

                # --------------------------------------------
                # RELEASE VIDEO WRITER
                # --------------------------------------------

                if video_writer is not None:

                    try:

                        video_writer.release()

                    except Exception:

                        pass

                video_writer = None

                recording = False

                video_is_valid = finalize_mp4(current_recording_temp_path, current_recording_path)
                print("Recording stopped.")
                if not video_is_valid:
                    print("ERROR: Recording was not finalised as a playable MP4.")

                # --------------------------------------------
                # SAVE USEFUL RECORDING (MINIMUM 1 MINUTE)
                # --------------------------------------------

                if (
                    video_is_valid
                    and current_recording_path is not None
                    and recording_duration >= MIN_RECORDING_SECONDS
                    and os.path.exists(current_recording_path)
                ):

                    try:

                        filename = (
                            os.path.basename(
                                current_recording_path
                            )
                        )

                        add_recording(
                            filename=filename,
                            filepath=current_recording_path,
                            duration=recording_duration
                        )

                        print(
                            "Recording saved to database:"
                        )

                        print(
                            f"  File: {filename}"
                        )

                        print(
                            f"  Duration: "
                            f"{recording_duration:.2f} seconds"
                        )

                    except Exception as e:

                        print(
                            "Recording database error:",
                            e
                        )

                elif current_recording_path is not None:

                    print(
                        "Recording too short. "
                        "Not added to database."
                    )

                # --------------------------------------------
                # RECORDING STOP EVENT
                # --------------------------------------------

                log_event(
                    "Recording stopped",
                    "recording_stop"
                )

                recording_start_time = None

                current_recording_path = None
                current_recording_temp_path = None

        # ====================================================
        # CAMERA STATUS
        # ====================================================

        if recording:

            camera_status = "RECORDING"

        elif motion_detected:

            camera_status = "MOTION DETECTED"

        else:

            camera_status = "ACTIVE"

        # ====================================================
        # UPDATE DASHBOARD STATUS
        # ====================================================

        # Update object monitor with newest frame
        object_monitor.update_frame(frame)
        obj_mon_status = object_monitor.get_status()

        # Log ONE alert when object goes missing, and ONE notification when restored
        if object_monitor.check_and_reset_alert_trigger():
            asset_title = obj_mon_status.get("object_name") or "Monitored asset"
            log_event(
                f"ALERT: Monitored asset '{asset_title}' missing from camera view",
                event_type="object_alert",
                event_key=f"obj_missing_{int(time.time())}"
            )
        if object_monitor.check_and_reset_restored_trigger():
            asset_title = obj_mon_status.get("object_name") or "Monitored asset"
            log_event(
                f"RESOLVED: Monitored asset '{asset_title}' restored to camera view",
                event_type="object_alert",
                event_key=f"obj_restored_{int(time.time())}"
            )

        send_dashboard_status(
            camera=camera_status,
            motion=motion_detected,
            face_detected=face_detected,
            people=detected_people,
            tamper=current_tamper_status,
            tampering=tamper_detected,
            recording=recording,
            siren=siren_manager.is_active(),
            object_monitor_status=obj_mon_status
        )

        # ====================================================
        # CAMERA OVERLAY
        # ====================================================

        if recording:

            cv2.putText(
                frame,
                "RECORDING",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "MONITORING",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2
            )

        # ====================================================
        # TAMPER STATUS, SIREN & OBJECT MONITOR OVERLAY
        # ====================================================

        tamper_color = (0, 0, 255) if tamper_detected else (255, 255, 0)
        cv2.putText(
            frame,
            f"Tamper: {current_tamper_status}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tamper_color,
            2
        )

        y_pos = 95
        if siren_manager.is_active():
            cv2.putText(
                frame,
                "SIREN ACTIVE - LENS COVERED",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )
            y_pos += 30

        # Monitored Object Visual Indicator on Camera Feed (Text only, NO bounding box)
        if obj_mon_status.get("has_object"):
            if obj_mon_status.get("alert"):
                obj_name_str = (obj_mon_status.get("object_name") or "OBJECT").upper()
                cv2.putText(
                    frame,
                    f"ALERT: {obj_name_str} MISSING!",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    f"Asset Protected: {obj_mon_status.get('object_name', 'Object')} (IN VIEW)",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        # ====================================================
        # UPDATE DASHBOARD FRAME
        # ====================================================

        try:
            now_for_dashboard = time.time()
            if now_for_dashboard - last_dashboard_frame_update >= DASHBOARD_FRAME_INTERVAL:
                update_frame(frame)
                last_dashboard_frame_update = now_for_dashboard
        except Exception as e:
            print("Dashboard frame update error:", e)

        # ====================================================
        # DISPLAY CAMERA
        # ====================================================

        cv2.imshow(
            "SENTRIX Smart Surveillance",
            frame
        )

        # ====================================================
        # KEYBOARD INPUT
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            print()

            print(
                "Q pressed."
            )

            print(
                "Stopping SENTRIX..."
            )

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    print(
        "Cleaning up..."
    )

    # --------------------------------------------------------
    # FINISH ACTIVE RECORDING
    # --------------------------------------------------------

    if video_writer is not None:

        try:

            video_writer.release()

        except Exception:

            pass

        video_writer = None

        video_is_valid = finalize_mp4(current_recording_temp_path, current_recording_path)

        recording_duration = 0

        if recording_start_time is not None:

            recording_duration = (
                time.time()
                - recording_start_time
            )

        # ----------------------------------------------------
        # SAVE ACTIVE RECORDING IF USEFUL
        # ----------------------------------------------------

        if (
            video_is_valid
            and current_recording_path is not None
            and recording_duration >= 2.0
            and os.path.exists(current_recording_path)
        ):

            try:

                filename = (
                    os.path.basename(
                        current_recording_path
                    )
                )

                add_recording(
                    filename=filename,
                    filepath=current_recording_path,
                    duration=recording_duration
                )

                print(
                    "Final recording saved to database:"
                )

                print(
                    f"  File: {filename}"
                )

                print(
                    f"  Duration: "
                    f"{recording_duration:.2f} seconds"
                )

            except Exception as e:

                print(
                    "Final recording database error:",
                    e
                )

        elif current_recording_path is not None:

            print(
                "Recording too short. "
                "Not added to database."
            )

        recording_start_time = None

        current_recording_path = None
        current_recording_temp_path = None

        recording = False

    # --------------------------------------------------------
    # STOP WORKER AND RELEASE CAMERA
    # --------------------------------------------------------

    face_worker_stop.set()
    try:
        face_worker_thread.join(timeout=1.0)
    except Exception:
        pass

    camera.release()

    cv2.destroyAllWindows()

    siren_manager.stop()
    object_monitor.stop()

    # ========================================================
    # FINAL DASHBOARD STATUS
    # ========================================================

    send_dashboard_status(
        camera="STOPPED",
        motion=False,
        face_detected=False,
        people=[],
        tamper="CAMERA OK",
        tampering=False,
        recording=False,
        siren=False
    )

    print()

    print("==============================")
    print("       SENTRIX STOPPED")
    print("==============================")
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    dashboard_thread = threading.Thread(
        target=start_dashboard,
        daemon=True
    )

    dashboard_thread.start()

    time.sleep(2)

    run_sentrix()