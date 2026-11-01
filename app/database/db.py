import os
import sqlite3
from datetime import datetime


# ============================================================
# SENTRIX DATABASE
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


def get_connection():
    """
    Create and return a connection to the SENTRIX SQLite database.
    """

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """
    Create all SENTRIX database tables if they do not already exist.
    """

    connection = get_connection()

    cursor = connection.cursor()

    # --------------------------------------------------------
    # REGISTERED PEOPLE
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

    # --------------------------------------------------------
    # OBJECT MISSING INTERVALS (FROM - TO)
    # --------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS object_missing_intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_seconds REAL DEFAULT 0,
            duration_str TEXT,
            status TEXT DEFAULT 'ACTIVE'
        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# PERSON FUNCTIONS
# ============================================================

def add_person(
    name,
    role="Person"
):
    """
    Add a registered person to SENTRIX.

    Returns:
        Person ID
    """

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
    """
    Return all registered people.
    """

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

    return [
        dict(row)
        for row in rows
    ]


def get_person(
    name
):
    """
    Find a person by name.
    """

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


def deactivate_person(
    name
):
    """
    Disable a registered person without deleting their history.
    """

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
# EVENT FUNCTIONS
# ============================================================

def add_event(
    event_type,
    message="",
    person_name=None,
    confidence=None
):
    """
    Store a security event.
    """

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


def get_recent_events(
    limit=100
):
    """
    Return the latest security events.
    """

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

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# RECORDING FUNCTIONS
# ============================================================

def add_recording(
    filename,
    filepath,
    duration=0
):
    """
    Store information about a recorded video.
    """

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
    """
    Return all stored recordings.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM recordings
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# TAMPER FUNCTIONS
# ============================================================

def add_tamper_event(
    tamper_type,
    brightness=None,
    blur_variance=None,
    histogram_correlation=None
):
    """
    Store camera tampering information.
    """

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


def get_tamper_events(
    limit=100
):
    """
    Return recent tamper events.
    """

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

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# QUERY & ANALYTICS HELPER FUNCTIONS
# ============================================================

def get_events_by_date(date_str):
    """
    Return all events recorded on date_str (YYYY-MM-DD), ordered chronologically.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT *
        FROM events
        WHERE timestamp LIKE ?
        ORDER BY id DESC
        """,
        (f"{date_str}%",)
    )
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


def get_events_by_type(event_type, limit=100, date_str=None):
    """
    Return events matching a specific event_type.
    """
    connection = get_connection()
    cursor = connection.cursor()
    if date_str:
        cursor.execute(
            """
            SELECT *
            FROM events
            WHERE event_type = ? AND timestamp LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (event_type, f"{date_str}%", limit)
        )
    else:
        cursor.execute(
            """
            SELECT *
            FROM events
            WHERE event_type = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (event_type, limit)
        )
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


def get_events_between(start_iso, end_iso, limit=200):
    """
    Return events within an ISO timestamp range.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT *
        FROM events
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (start_iso, end_iso, limit)
    )
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


def get_hourly_activity(date_str):
    """
    Return an array of 24 integers representing event counts for each hour (0..23) on date_str.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT strftime('%H', timestamp) AS hour_str, COUNT(*) AS count
        FROM events
        WHERE timestamp LIKE ?
        GROUP BY hour_str
        ORDER BY hour_str ASC
        """,
        (f"{date_str}%",)
    )
    rows = cursor.fetchall()
    connection.close()

    hourly = [0] * 24
    for row in rows:
        try:
            h = int(row["hour_str"])
            if 0 <= h < 24:
                hourly[h] = row["count"]
        except Exception:
            pass
    return hourly


def get_event_type_distribution(date_str=None):
    """
    Return a dictionary mapping event_type to its count.
    """
    connection = get_connection()
    cursor = connection.cursor()
    if date_str:
        cursor.execute(
            """
            SELECT event_type, COUNT(*) AS count
            FROM events
            WHERE timestamp LIKE ?
            GROUP BY event_type
            ORDER BY count DESC
            """,
            (f"{date_str}%",)
        )
    else:
        cursor.execute(
            """
            SELECT event_type, COUNT(*) AS count
            FROM events
            GROUP BY event_type
            ORDER BY count DESC
            """
        )
    rows = cursor.fetchall()
    connection.close()
    return {row["event_type"]: row["count"] for row in rows}


def get_person_activity(person_name=None, date_str=None):
    """
    Return detection events and summary statistics for a registered person.
    """
    connection = get_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM events WHERE event_type = 'face_recognized'"
    params = []

    if person_name:
        query += " AND person_name LIKE ?"
        params.append(f"%{person_name}%")

    if date_str:
        query += " AND timestamp LIKE ?"
        params.append(f"{date_str}%")

    query += " ORDER BY id DESC LIMIT 100"

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    connection.close()
    return [dict(row) for row in rows]


def get_tamper_summary(date_str=None):
    """
    Return tamper event counts grouped by tamper_type.
    """
    connection = get_connection()
    cursor = connection.cursor()
    if date_str:
        cursor.execute(
            """
            SELECT tamper_type, COUNT(*) as count
            FROM tamper_events
            WHERE timestamp LIKE ?
            GROUP BY tamper_type
            """,
            (f"{date_str}%",)
        )
    else:
        cursor.execute(
            """
            SELECT tamper_type, COUNT(*) as count
            FROM tamper_events
            GROUP BY tamper_type
            """
        )
    rows = cursor.fetchall()
    connection.close()
    return {row["tamper_type"]: row["count"] for row in rows}


def get_recordings_summary(date_str=None):
    """
    Return count, total duration, and longest recording.
    """
    connection = get_connection()
    cursor = connection.cursor()
    if date_str:
        cursor.execute(
            """
            SELECT COUNT(*) AS total_count, COALESCE(SUM(duration), 0) AS total_duration,
                   COALESCE(MAX(duration), 0) AS max_duration
            FROM recordings
            WHERE created_at LIKE ?
            """,
            (f"{date_str}%",)
        )
    else:
        cursor.execute(
            """
            SELECT COUNT(*) AS total_count, COALESCE(SUM(duration), 0) AS total_duration,
                   COALESCE(MAX(duration), 0) AS max_duration
            FROM recordings
            """
        )
    row = cursor.fetchone()
    connection.close()
    return dict(row) if row else {"total_count": 0, "total_duration": 0, "max_duration": 0}


# ============================================================
# MONITORED OBJECT MISSING INTERVALS
# ============================================================

def start_missing_interval(asset_name, start_time):
    """
    Record the start of a missing object incident. Returns the new interval ID.
    """
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO object_missing_intervals (asset_name, start_time, status)
        VALUES (?, ?, 'CURRENTLY MISSING')
        """,
        (asset_name or "Monitored Asset", start_time)
    )
    interval_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return interval_id


def end_missing_interval(interval_id, end_time, duration_seconds=0, duration_str=""):
    """
    Mark a missing object incident as restored with final end time and duration.
    """
    if not interval_id:
        return
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        UPDATE object_missing_intervals
        SET end_time = ?, duration_seconds = ?, duration_str = ?, status = 'RESTORED'
        WHERE id = ?
        """,
        (end_time, float(duration_seconds), duration_str, interval_id)
    )
    connection.commit()
    connection.close()


def get_missing_intervals(date_str=None):
    """
    Retrieve recorded missing object intervals, optionally filtered by date.
    """
    connection = get_connection()
    cursor = connection.cursor()
    if date_str:
        cursor.execute(
            """
            SELECT id, asset_name, start_time, end_time, duration_seconds, duration_str, status
            FROM object_missing_intervals
            WHERE start_time LIKE ?
            ORDER BY id DESC
            """,
            (f"{date_str}%",)
        )
    else:
        cursor.execute(
            """
            SELECT id, asset_name, start_time, end_time, duration_seconds, duration_str, status
            FROM object_missing_intervals
            ORDER BY id DESC
            """
        )
    rows = cursor.fetchall()
    connection.close()
    return [dict(r) for r in rows]


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

initialize_database()