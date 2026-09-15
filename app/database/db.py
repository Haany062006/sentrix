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
# DATABASE INITIALIZATION
# ============================================================

initialize_database()