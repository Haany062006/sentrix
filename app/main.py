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
EVENT_COOLDOWN = 5
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
    histogram_correlation=None
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

    # Event cooldown
    if event_key in last_event_times:

        elapsed = (
            current_time -
            last_event_times[event_key]
        )

        if elapsed < EVENT_COOLDOWN:
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

        message_lower = message.lower()

        if "motion detected" in message_lower:

            event_type = "motion"

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
    recording=False
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

    recording_start_time = None

    # ========================================================
    # CURRENT TAMPER STATUS
    # ========================================================

    current_tamper_status = "CAMERA OK"

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
        # TAMPER DETECTION
        # ====================================================

        tamper_detected = False

        current_tamper_status = "CAMERA OK"

        tamper_brightness = None

        tamper_blur_variance = None

        tamper_histogram_correlation = None

        if tamper_detector is not None:

            try:

                tamper_result = (
                    tamper_detector.detect(
                        frame
                    )
                )

                if isinstance(
                    tamper_result,
                    dict
                ):

                    tamper_detected = bool(
                        tamper_result.get(
                            "tampering",
                            False
                        )
                    )

                    current_tamper_status = (
                        tamper_result.get(
                            "status",
                            "CAMERA OK"
                        )
                    )

                    tamper_brightness = (
                        tamper_result.get(
                            "brightness"
                        )
                    )

                    tamper_blur_variance = (
                        tamper_result.get(
                            "blur_variance"
                        )
                    )

                    tamper_histogram_correlation = (
                        tamper_result.get(
                            "histogram_correlation"
                        )
                    )

                else:

                    current_tamper_status = str(
                        tamper_result
                    )

                # --------------------------------------------
                # TAMPER EVENT
                # --------------------------------------------

                if tamper_detected:

                    log_event(
                        (
                            f"Tamper detected: "
                            f"{current_tamper_status}"
                        ),
                        (
                            f"tamper_"
                            f"{current_tamper_status}"
                        ),
                        tamper_type=(
                            current_tamper_status
                        ),
                        brightness=(
                            tamper_brightness
                        ),
                        blur_variance=(
                            tamper_blur_variance
                        ),
                        histogram_correlation=(
                            tamper_histogram_correlation
                        )
                    )

            except Exception as e:

                print(
                    "Tamper detection error:",
                    e
                )

        # ====================================================
        # FACE DETECTION
        # ====================================================

        try:

            detections = (
                face_detector.detect_faces(
                    rgb_frame
                )
            )

        except Exception as e:

            print(
                "Face detection error:",
                e
            )

            detections = []

        # ====================================================
        # PERSON STATUS
        # ====================================================

        face_detected = False

        detected_people = []

        # ====================================================
        # PROCESS FACES
        # ====================================================

        for detection in detections:

            try:

                # --------------------------------------------
                # MTCNN CONFIDENCE
                # --------------------------------------------

                detection_confidence = (
                    detection.get(
                        "confidence",
                        0
                    )
                )

                if detection_confidence < 0.90:

                    continue

                # --------------------------------------------
                # FACE BOUNDING BOX
                # --------------------------------------------

                x, y, w, h = (
                    detection["box"]
                )

                x = max(
                    0,
                    x
                )

                y = max(
                    0,
                    y
                )

                x2 = min(
                    frame_width,
                    x + w
                )

                y2 = min(
                    frame_height,
                    y + h
                )

                if x2 <= x or y2 <= y:

                    continue

                # --------------------------------------------
                # FACE IMAGE
                # --------------------------------------------

                face_image = frame[
                    y:y2,
                    x:x2
                ]

                if face_image.size == 0:

                    continue

                face_detected = True

                # --------------------------------------------
                # FACE RECOGNITION
                # --------------------------------------------

                try:

                    name, score = (
                        recognize_face(
                            face_image,
                            embedder,
                            registered_faces
                        )
                    )

                except Exception as e:

                    print(
                        "Face recognition error:",
                        e
                    )

                    name = "ERROR"

                    score = 0.0

                # --------------------------------------------
                # UNKNOWN PERSON
                # --------------------------------------------

                if name == "UNKNOWN":

                    person_display_name = (
                        "Unknown Person"
                    )

                    if (
                        person_display_name
                        not in detected_people
                    ):

                        detected_people.append(
                            person_display_name
                        )

                    box_color = (
                        0,
                        0,
                        255
                    )

                    label = (
                        f"Unknown "
                        f"({score:.2f})"
                    )

                    log_event(
                        (
                            f"Unknown person detected "
                            f"({score:.2f})"
                        ),
                        "unknown_person",
                        confidence=score
                    )

                # --------------------------------------------
                # RECOGNITION ERROR
                # --------------------------------------------

                elif name == "ERROR":

                    person_display_name = (
                        "Recognition Error"
                    )

                    if (
                        person_display_name
                        not in detected_people
                    ):

                        detected_people.append(
                            person_display_name
                        )

                    box_color = (
                        0,
                        255,
                        255
                    )

                    label = (
                        "Recognition Error"
                    )

                # --------------------------------------------
                # KNOWN PERSON
                # --------------------------------------------

                else:

                    person_display_name = (
                        name
                    )

                    if (
                        person_display_name
                        not in detected_people
                    ):

                        detected_people.append(
                            person_display_name
                        )

                    box_color = (
                        0,
                        255,
                        0
                    )

                    label = (
                        f"{name} "
                        f"({score:.2f})"
                    )

                    log_event(
                        (
                            f"{name} recognized "
                            f"({score:.2f})"
                        ),
                        f"recognized_{name}",
                        person_name=name,
                        confidence=score
                    )

                # --------------------------------------------
                # DRAW FACE BOX
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x2, y2),
                    box_color,
                    2
                )

                # --------------------------------------------
                # DRAW NAME
                # --------------------------------------------

                cv2.putText(
                    frame,
                    label,
                    (
                        x,
                        max(
                            30,
                            y - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2
                )

            except Exception as e:

                print(
                    "Face processing error:",
                    e
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

            current_recording_path = (
                os.path.join(
                    OUTPUT_FOLDER,
                    (
                        f"SENTRIX_"
                        f"{timestamp}.mp4"
                    )
                )
            )

            fourcc = (
                cv2.VideoWriter_fourcc(
                    *"mp4v"
                )
            )

            video_writer = (
                cv2.VideoWriter(
                    current_recording_path,
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

            if (
                elapsed_since_motion
                >= POST_MOTION_SECONDS
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

                print(
                    "Recording stopped."
                )

                # --------------------------------------------
                # SAVE USEFUL RECORDING
                # --------------------------------------------

                if (
                    current_recording_path is not None
                    and recording_duration >= 2.0
                    and os.path.exists(
                        current_recording_path
                    )
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

        send_dashboard_status(
            camera=camera_status,
            motion=motion_detected,
            face_detected=face_detected,
            people=detected_people,
            tamper=current_tamper_status,
            tampering=tamper_detected,
            recording=recording
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
        # TAMPER STATUS
        # ====================================================

        cv2.putText(
            frame,
            f"Tamper: {current_tamper_status}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

        # ====================================================
        # UPDATE DASHBOARD FRAME
        # ====================================================

        try:

            update_frame(
                frame
            )

        except Exception as e:

            print(
                "Dashboard frame update error:",
                e
            )

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
            current_recording_path is not None
            and recording_duration >= 2.0
            and os.path.exists(
                current_recording_path
            )
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

        recording = False

    # --------------------------------------------------------
    # RELEASE CAMERA
    # --------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()

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
        recording=False
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