import os
import sqlite3
from datetime import datetime, timedelta


# ============================================================
# SENTRIX DATABASE PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

DB_PATH = os.path.join(
    DATA_FOLDER,
    "sentrix.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    os.makedirs(DATA_FOLDER, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # REGISTERED PERSONS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            role TEXT DEFAULT 'Person',
            registered_at TEXT NOT NULL,
            active INTEGER DEFAULT 1
        )
    """)

    # --------------------------------------------------------
    # SECURITY EVENTS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            message TEXT,
            person_name TEXT,
            confidence REAL,
            timestamp TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # RECORDINGS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            filepath TEXT,
            created_at TEXT NOT NULL,
            duration REAL DEFAULT 0
        )
    """)

    # --------------------------------------------------------
    # TAMPER EVENTS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tamper_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tamper_type TEXT NOT NULL,
            brightness REAL,
            blur_variance REAL,
            histogram_correlation REAL,
            timestamp TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# ============================================================
# PERSON MANAGEMENT
# ============================================================

def add_person(name, role="Person"):

    connection = get_connection()
    cursor = connection.cursor()

    registered_at = datetime.now().isoformat(
        timespec="seconds"
    )

    try:

        cursor.execute(
            """
            INSERT INTO persons
            (
                name,
                role,
                registered_at,
                active
            )
            VALUES (?, ?, ?, 1)
            """,
            (
                name,
                role,
                registered_at
            )
        )

        connection.commit()

        person_id = cursor.lastrowid

    except sqlite3.IntegrityError:

        cursor.execute(
            """
            SELECT id
            FROM persons
            WHERE name = ?
            """,
            (name,)
        )

        row = cursor.fetchone()

        person_id = row["id"] if row else None

    connection.close()

    return person_id


def get_all_persons():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            role,
            registered_at,
            active
        FROM persons
        ORDER BY name
    """)

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_person(name):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM persons
        WHERE name = ?
        """,
        (name,)
    )

    row = cursor.fetchone()

    connection.close()

    if row:
        return dict(row)

    return None


def deactivate_person(name):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE persons
        SET active = 0
        WHERE name = ?
        """,
        (name,)
    )

    connection.commit()
    connection.close()


# ============================================================
# EVENT MANAGEMENT
# ============================================================

def add_event(
    event_type,
    message="",
    person_name=None,
    confidence=None
):

    connection = get_connection()
    cursor = connection.cursor()

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    cursor.execute(
        """
        INSERT INTO events
        (
            event_type,
            message,
            person_name,
            confidence,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            event_type,
            message,
            person_name,
            confidence,
            timestamp
        )
    )

    connection.commit()
    connection.close()


def get_recent_events(limit=100):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# COMPLETE EVENTS FOR A PARTICULAR DATE
# ============================================================

def get_events_for_date(
    selected_date,
    until=None
):
    """
    Returns ALL events for the selected date.

    selected_date:
        YYYY-MM-DD

    until:
        Optional datetime.
        If supplied, events after this time are excluded.

    Unlike get_recent_events(), this does NOT have
    a 100-event limit.
    """

    connection = get_connection()
    cursor = connection.cursor()

    start_timestamp = f"{selected_date}T00:00:00"

    if until is None:

        end_timestamp = f"{selected_date}T23:59:59"

    else:

        if isinstance(until, datetime):
            end_timestamp = until.isoformat(
                timespec="seconds"
            )

        else:
            end_timestamp = str(until)

    cursor.execute(
        """
        SELECT *
        FROM events
        WHERE timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp ASC, id ASC
        """,
        (
            start_timestamp,
            end_timestamp
        )
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# RECORDING MANAGEMENT
# ============================================================

def add_recording(
    filename,
    filepath,
    duration=0
):

    connection = get_connection()
    cursor = connection.cursor()

    created_at = datetime.now().isoformat(
        timespec="seconds"
    )

    cursor.execute(
        """
        INSERT OR IGNORE INTO recordings
        (
            filename,
            filepath,
            created_at,
            duration
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            filename,
            filepath,
            created_at,
            duration
        )
    )

    connection.commit()
    connection.close()


def get_recordings():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM recordings
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# COMPLETE RECORDINGS FOR A PARTICULAR DATE
# ============================================================

def get_recordings_for_date(
    selected_date,
    until=None
):
    """
    Returns ALL recordings created on the selected date.

    This does not remove records whose video file has already
    been deleted by the 15-day retention policy.
    """

    connection = get_connection()
    cursor = connection.cursor()

    start_timestamp = f"{selected_date}T00:00:00"

    if until is None:

        end_timestamp = f"{selected_date}T23:59:59"

    else:

        if isinstance(until, datetime):
            end_timestamp = until.isoformat(
                timespec="seconds"
            )

        else:
            end_timestamp = str(until)

    cursor.execute(
        """
        SELECT *
        FROM recordings
        WHERE created_at >= ?
        AND created_at <= ?
        ORDER BY created_at ASC, id ASC
        """,
        (
            start_timestamp,
            end_timestamp
        )
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# 15-DAY RECORDING RETENTION
# ============================================================

def cleanup_old_recording_files(days=15):
    """
    Deletes physical recording files older than the specified
    number of days.

    IMPORTANT:
    The database record is NEVER deleted.

    Example:

        Video file:
            recordings/SENTRIX_20260901_103000.mp4
            -> deleted after 15 days

        Database record:
            remains permanently

    This allows SENTRIX reports to retain historical evidence
    information even after the physical video is removed.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cutoff_time = datetime.now() - timedelta(days=days)

    cursor.execute("""
        SELECT
            id,
            filename,
            filepath,
            created_at
        FROM recordings
        WHERE created_at IS NOT NULL
    """)

    rows = cursor.fetchall()

    deleted_files = []
    missing_files = []
    failed_files = []

    for row in rows:

        filename = row["filename"]
        filepath = row["filepath"]
        created_at = row["created_at"]

        # ----------------------------------------------------
        # Convert stored timestamp
        # ----------------------------------------------------

        try:

            recording_time = datetime.fromisoformat(
                created_at
            )

        except (
            ValueError,
            TypeError
        ):

            print(
                f"[SENTRIX] Invalid recording timestamp: "
                f"{filename}"
            )

            continue

        # ----------------------------------------------------
        # Keep recordings within retention period
        # ----------------------------------------------------

        if recording_time >= cutoff_time:
            continue

        # ----------------------------------------------------
        # Determine actual file path
        # ----------------------------------------------------

        actual_path = filepath

        if not actual_path:

            actual_path = os.path.join(
                BASE_DIR,
                "recordings",
                filename
            )

        # ----------------------------------------------------
        # Delete physical video
        # ----------------------------------------------------

        if os.path.exists(actual_path):

            try:

                os.remove(actual_path)

                deleted_files.append(filename)

                print(
                    f"[SENTRIX] Deleted old recording: "
                    f"{filename}"
                )

            except OSError as error:

                failed_files.append(filename)

                print(
                    f"[SENTRIX] Could not delete "
                    f"{filename}: {error}"
                )

        else:

            # File is already missing.
            # Database record is still preserved.

            missing_files.append(filename)

    connection.close()

    print(
        "[SENTRIX] Recording cleanup completed."
    )

    print(
        f"[SENTRIX] Deleted: "
        f"{len(deleted_files)}"
    )

    print(
        f"[SENTRIX] Already missing: "
        f"{len(missing_files)}"
    )

    print(
        f"[SENTRIX] Failed: "
        f"{len(failed_files)}"
    )

    return {
        "deleted": deleted_files,
        "missing": missing_files,
        "failed": failed_files
    }


# ============================================================
# TAMPER EVENT MANAGEMENT
# ============================================================

def add_tamper_event(
    tamper_type,
    brightness=None,
    blur_variance=None,
    histogram_correlation=None
):

    connection = get_connection()
    cursor = connection.cursor()

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    cursor.execute(
        """
        INSERT INTO tamper_events
        (
            tamper_type,
            brightness,
            blur_variance,
            histogram_correlation,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            tamper_type,
            brightness,
            blur_variance,
            histogram_correlation,
            timestamp
        )
    )

    connection.commit()
    connection.close()


def get_tamper_events(limit=100):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM tamper_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# COMPLETE TAMPER EVENTS FOR A PARTICULAR DATE
# ============================================================

def get_tamper_events_for_date(
    selected_date,
    until=None
):
    """
    Returns ALL tamper events for the selected date.

    No 100-event limit.
    """

    connection = get_connection()
    cursor = connection.cursor()

    start_timestamp = f"{selected_date}T00:00:00"

    if until is None:

        end_timestamp = f"{selected_date}T23:59:59"

    else:

        if isinstance(until, datetime):
            end_timestamp = until.isoformat(
                timespec="seconds"
            )

        else:
            end_timestamp = str(until)

    cursor.execute(
        """
        SELECT *
        FROM tamper_events
        WHERE timestamp >= ?
        AND timestamp <= ?
        ORDER BY timestamp ASC, id ASC
        """,
        (
            start_timestamp,
            end_timestamp
        )
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()