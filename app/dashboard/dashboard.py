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

import cv2
import numpy as np

from app.reports.generator import generate_daily_report
from app.query.engine import query_engine
from app.tamper.siren import siren_manager
from app.object_monitor import object_monitor
from app.reports.visualizer import report_visualizer


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
    "recording": False,
    "siren": False,
    "object_monitor": object_monitor.get_status()
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

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

RECORDINGS_FOLDER = os.path.join(
    BASE_DIR,
    "recordings"
)

REPORTS_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)


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
    selected_date = request.args.get("date")

    # Fetch database recordings for stored durations
    db_records = {}
    try:
        from app.database.db import get_recordings
        for rec in get_recordings():
            fname = rec.get("filename")
            if fname:
                db_records[fname] = rec
    except Exception as exc:
        print("Database recordings query error:", exc)

    files = []
    by_date = {}

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
        file_size = os.path.getsize(filepath)

        # Parse date and time from filename: SENTRIX_YYYYMMDD_HHMMSS.mp4
        date_str = ""
        time_str = ""
        clean_name = os.path.splitext(filename)[0]
        parts = clean_name.split("_")
        if len(parts) >= 3 and len(parts[1]) == 8:
            raw_date = parts[1]
            raw_time = parts[2]
            date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            if len(raw_time) >= 6:
                time_str = f"{raw_time[:2]}:{raw_time[2:4]}:{raw_time[4:6]}"

        if not date_str:
            date_str = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d")
        if not time_str:
            time_str = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%H:%M:%S")

        # Filter by date if specified and not 'ALL'
        if selected_date and selected_date != "ALL" and date_str != selected_date:
            continue

        db_rec = db_records.get(filename, {})
        duration = db_rec.get("duration", 0.0)

        # Duration formatting (recording minimum is 1 min)
        if duration and duration > 0:
            m = int(duration // 60)
            s = int(duration % 60)
            duration_formatted = f"{m}m {s:02d}s" if m > 0 else f"{s}s"
        else:
            duration_formatted = "≥ 1 min"

        entry = {
            "name": filename,
            "size": file_size,
            "date": date_str,
            "time": time_str,
            "duration": round(duration, 1) if duration else 0,
            "duration_formatted": duration_formatted,
            "url": f"/recordings/{filename}"
        }
        files.append(entry)

    # Newest first
    files.sort(
        key=lambda x: x["name"],
        reverse=True
    )

    # Group date-wise
    for f in files:
        d = f["date"]
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(f)

    dates = sorted(list(by_date.keys()), reverse=True)

    return jsonify({
        "success": True,
        "total": len(files),
        "dates": dates,
        "by_date": by_date,
        "recordings": files
    })


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
# QUERY ENGINE API (EXAMINER RECOMMENDATION 1)
# ============================================================

@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Process natural language or structured surveillance queries.
    """
    data = request.get_json(silent=True) or {}
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({
            "success": False,
            "error": "Query cannot be empty."
        }), 400

    try:
        result = query_engine.query(user_query)
        return jsonify(result)
    except Exception as exc:
        print("Query API error:", exc)
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@app.route("/api/query/suggestions", methods=["GET"])
def api_query_suggestions():
    """
    Return popular suggested query prompts for quick execution.
    """
    suggestions = [
        "When was the object last there?",
        "When was the object missing?",
        "Security summary for today",
        "Who was detected today?",
        "Was there any camera tampering?",
        "How many recordings were made?"
    ]
    return jsonify({
        "success": True,
        "suggestions": suggestions
    })


# ============================================================
# SIREN DETERRENT API (EXAMINER RECOMMENDATION 4)
# ============================================================

@app.route("/api/siren/status", methods=["GET"])
def api_siren_status():
    """
    Get current siren deterrent alarm status.
    """
    return jsonify({
        "success": True,
        "siren": siren_manager.get_status()
    })


@app.route("/api/siren/trigger", methods=["POST"])
def api_siren_trigger():
    """
    Manually trigger or test the deterrent siren.
    """
    data = request.get_json(silent=True) or {}
    duration = float(data.get("duration", 10.0))
    reason = str(data.get("reason", "MANUAL SIREN TEST"))
    siren_manager.trigger(reason=reason, duration=duration)
    return jsonify({
        "success": True,
        "message": f"Siren activated ({reason})",
        "siren": siren_manager.get_status()
    })


@app.route("/api/siren/mute", methods=["POST"])
def api_siren_mute():
    """
    Mute or silence the active siren deterrent.
    """
    siren_manager.stop()
    return jsonify({
        "success": True,
        "message": "Siren silenced",
        "siren": siren_manager.get_status()
    })


# ============================================================
# MONITORED OBJECT API (UPLOAD OBJECT TO MONITOR)
# ============================================================

@app.route("/api/object-monitor/status", methods=["GET"])
def api_object_monitor_status():
    """
    Get current status of the user-monitored object.
    """
    return jsonify({
        "success": True,
        "status": object_monitor.get_status()
    })


@app.route("/api/object-monitor/upload", methods=["POST"])
def api_object_monitor_upload():
    """
    Upload an image of an object to monitor in the surveillance camera view.
    """
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return jsonify({
            "success": False,
            "error": "No image file uploaded."
        }), 400

    try:
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return jsonify({
                "success": False,
                "error": "Could not decode uploaded image."
            }), 400

        object_name = file.filename.rsplit(".", 1)[0].replace("_", " ").title()
        ok = object_monitor.set_object_image(img, name=object_name)

        if ok:
            return jsonify({
                "success": True,
                "message": f"Object '{object_name}' registered for active monitoring.",
                "status": object_monitor.get_status()
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to set object image."
            }), 500

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/object-monitor/clear", methods=["POST"])
def api_object_monitor_clear():
    """
    Stop monitoring and clear the uploaded object.
    """
    object_monitor.clear_object()
    return jsonify({
        "success": True,
        "message": "Monitored object cleared.",
        "status": object_monitor.get_status()
    })


@app.route("/api/object-monitor/image", methods=["GET"])
def api_object_monitor_image():
    """
    Serve the currently monitored object image.
    """
    if not object_monitor.has_object or not os.path.exists(object_monitor.storage_path):
        return jsonify({"error": "No monitored object set"}), 404

    try:
        with open(object_monitor.storage_path, "rb") as f:
            img_data = f.read()
        return Response(img_data, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )