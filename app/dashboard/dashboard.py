import os
import threading

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    send_from_directory,
    request
)

from app.reports.generator import generate_daily_report


app = Flask(__name__)


# ============================================================
# SENTRIX SHARED DATA
# ============================================================

latest_frame = None

status_data = {
    "camera": "STARTING",
    "motion": False,
    "face_detected": False,
    "people": [],
    "tamper": "CAMERA OK",
    "tampering": False,
    "recording": False
}

events = []


# ============================================================
# THREAD LOCKS
# ============================================================

frame_lock = threading.Lock()
status_lock = threading.Lock()
events_lock = threading.Lock()


# ============================================================
# FOLDERS
# ============================================================

RECORDINGS_FOLDER = "recordings"
REPORTS_FOLDER = "reports"


# ============================================================
# CAMERA FRAME
# ============================================================

def update_frame(frame):

    global latest_frame

    with frame_lock:
        latest_frame = frame.copy()


def get_frame():

    with frame_lock:

        if latest_frame is None:
            return None

        return latest_frame.copy()


# ============================================================
# STATUS
# ============================================================

def update_status(data):

    global status_data

    with status_lock:
        status_data = data.copy()


def get_status():

    with status_lock:
        return status_data.copy()


# ============================================================
# EVENTS
# ============================================================

def add_event(message):

    with events_lock:

        events.insert(
            0,
            message
        )

        # Keep only latest 20 events
        if len(events) > 20:

            events.pop()


def get_events():

    with events_lock:
        return list(events)


# ============================================================
# LIVE VIDEO GENERATOR
# ============================================================

def generate_video():

    import cv2

    while True:

        frame = get_frame()

        if frame is None:
            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            continue

        frame_bytes = encoded.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "index.html"
    )


# ============================================================
# LIVE CAMERA
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_video(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# STATUS API
# ============================================================

@app.route("/api/status")
def api_status():

    return jsonify(
        get_status()
    )


# ============================================================
# EVENTS API
# ============================================================

@app.route("/api/events")
def api_events():

    return jsonify(
        get_events()
    )


# ============================================================
# RECORDING FILE
# ============================================================

@app.route("/recordings/<path:filename>")
def recordings(filename):

    return send_from_directory(
        RECORDINGS_FOLDER,
        filename
    )


# ============================================================
# RECORDINGS API
# ============================================================

@app.route("/api/recordings")
def api_recordings():

    os.makedirs(
        RECORDINGS_FOLDER,
        exist_ok=True
    )

    files = []

    for filename in os.listdir(
        RECORDINGS_FOLDER
    ):

        if not filename.lower().endswith(
            (
                ".avi",
                ".mp4",
                ".mov",
                ".mkv"
            )
        ):
            continue

        filepath = os.path.join(
            RECORDINGS_FOLDER,
            filename
        )

        files.append({
            "name": filename,
            "size": os.path.getsize(filepath)
        })

    # Newest first
    files.sort(
        key=lambda x: x["name"],
        reverse=True
    )

    return jsonify(files)


# ============================================================
# GENERATE DAILY SECURITY REPORT
# ============================================================

@app.route(
    "/api/generate-report",
    methods=["POST"]
)
def generate_report():

    # --------------------------------
    # READ REQUEST
    # --------------------------------

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "success": False,
            "error": "Invalid request."
        }), 400

    selected_date = data.get(
        "date"
    )

    if not selected_date:

        return jsonify({
            "success": False,
            "error": "Date is required."
        }), 400

    # --------------------------------
    # GET EVENTS
    # --------------------------------

    current_events = get_events()

    # --------------------------------
    # GET RECORDINGS
    # --------------------------------

    recordings = []

    os.makedirs(
        RECORDINGS_FOLDER,
        exist_ok=True
    )

    for filename in os.listdir(
        RECORDINGS_FOLDER
    ):

        if not filename.lower().endswith(
            (
                ".avi",
                ".mp4",
                ".mov",
                ".mkv"
            )
        ):
            continue

        filepath = os.path.join(
            RECORDINGS_FOLDER,
            filename
        )

        recordings.append({
            "name": filename,
            "size": os.path.getsize(filepath)
        })

    # --------------------------------
    # GENERATE PDF
    # --------------------------------

    try:

        filepath = generate_daily_report(
            selected_date,
            current_events,
            recordings
        )

        return jsonify({
            "success": True,
            "filename": os.path.basename(
                filepath
            )
        })

    except Exception as error:

        print(
            f"Report generation error: {error}"
        )

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# REPORT FILE
# ============================================================

@app.route(
    "/reports/<path:filename>"
)
def reports(filename):

    return send_from_directory(
        REPORTS_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# START DASHBOARD
# ============================================================

def start_dashboard():

    print(
        "\n--------------------------------"
    )

    print(
        "SENTRIX Dashboard:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "--------------------------------\n"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )