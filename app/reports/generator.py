
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

REPORTS_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORTS_FOLDER,
    exist_ok=True
)


# ============================================================
# CONSTANTS
# ============================================================

PAGE_WIDTH, PAGE_HEIGHT = A4

DARK = colors.HexColor("#101820")
DARK_BLUE = colors.HexColor("#162A43")
BLUE = colors.HexColor("#246BCE")
LIGHT_BLUE = colors.HexColor("#EAF2FF")

GREEN = colors.HexColor("#198754")
LIGHT_GREEN = colors.HexColor("#E8F7EE")

ORANGE = colors.HexColor("#D97706")
LIGHT_ORANGE = colors.HexColor("#FFF4DF")

RED = colors.HexColor("#C62828")
LIGHT_RED = colors.HexColor("#FDECEC")

GREY = colors.HexColor("#6B7280")
LIGHT_GREY = colors.HexColor("#F4F6F8")

BORDER = colors.HexColor("#D8DEE6")


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value, default="—"):
    """
    Safely convert any value into displayable text.
    """

    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text


def parse_timestamp(value):
    """
    Convert SENTRIX timestamps into datetime.

    Supports:
        YYYY-MM-DDTHH:MM:SS
        YYYY-MM-DD HH:MM:SS
        YYYY-MM-DDTHH:MM
        YYYY-MM-DD HH:MM
    """

    if not value:
        return None

    if isinstance(value, datetime):
        return value

    value = str(value).strip()

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(
                value,
                fmt
            )
        except ValueError:
            continue

    return None


def format_time(value):
    """
    Convert timestamp to HH:MM:SS.
    """

    dt = parse_timestamp(value)

    if dt:
        return dt.strftime("%H:%M:%S")

    return safe_text(value)


def format_datetime(value):
    """
    Convert timestamp to readable date/time.
    """

    dt = parse_timestamp(value)

    if dt:
        return dt.strftime(
            "%d %b %Y, %H:%M:%S"
        )

    return safe_text(value)


def format_duration(seconds):
    """
    Convert seconds into readable duration.
    """

    try:
        seconds = float(seconds or 0)
    except (TypeError, ValueError):
        return "0 sec"

    if seconds < 60:
        return f"{seconds:.1f} sec"

    minutes = int(seconds // 60)
    remaining = int(seconds % 60)

    return f"{minutes} min {remaining} sec"


def confidence_text(value):
    """
    Format face-recognition confidence.
    """

    if value is None:
        return "Not recorded"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return safe_text(value)

    if value <= 1:
        value *= 100

    return f"{value:.1f}%"


def paragraph(text, style):
    """
    Safely create a ReportLab Paragraph.
    """

    text = safe_text(
        text
    )

    # Escape ampersands so ReportLab XML does not break.
    text = text.replace(
        "&",
        "&amp;"
    )

    return Paragraph(
        text,
        style
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_report_items(items, item_type="event"):
    """
    Convert report input into a list of dictionaries.

    This prevents:
        'str' object has no attribute 'get'

    from crashing the report generator.

    Accepted input:
        - list of dictionaries
        - list of strings
        - tuple
        - single dictionary
        - single string
        - None
    """

    if items is None:
        return []

    if isinstance(items, dict):
        items = [items]

    elif isinstance(items, str):
        items = [items]

    elif not isinstance(items, (list, tuple)):
        items = [items]

    normalized = []

    for item in items:

        # ----------------------------------------------------
        # Already a dictionary
        # ----------------------------------------------------

        if isinstance(item, dict):
            normalized.append(item)
            continue

        # ----------------------------------------------------
        # String item
        # ----------------------------------------------------

        if isinstance(item, str):

            raw = item.strip()

            if not raw:
                continue

            timestamp = None
            message = raw

            # Try to extract timestamp from the beginning.
            possible_timestamp = raw[:19]

            parsed = parse_timestamp(
                possible_timestamp
            )

            if parsed:
                timestamp = possible_timestamp

                # Remove timestamp separators.
                message = raw[19:].strip()

                while message.startswith("-"):
                    message = message[1:].strip()

            lower_message = message.lower()

            # Determine event type.
            if "tamper" in lower_message:
                event_type = "tamper"

            elif (
                "unknown person" in lower_message
                or "unknown" in lower_message
            ):
                event_type = "unknown_person"

            elif (
                "recognized" in lower_message
                or "recognised" in lower_message
            ):
                event_type = "face_recognized"

            elif "motion" in lower_message:
                event_type = "motion"

            elif "recording" in lower_message:
                event_type = "recording"

            else:
                event_type = item_type

            normalized.append(
                {
                    "event_type": event_type,
                    "message": message,
                    "person_name": None,
                    "confidence": None,
                    "timestamp": timestamp,
                }
            )

            continue

        # ----------------------------------------------------
        # Unknown object
        # ----------------------------------------------------

        normalized.append(
            {
                "event_type": item_type,
                "message": str(item),
                "person_name": None,
                "confidence": None,
                "timestamp": None,
            }
        )

    return normalized


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

def classify_event(event):
    """
    Determine readable event category.
    """

    if not isinstance(event, dict):
        return "SYSTEM"

    event_type = str(
        event.get(
            "event_type",
            ""
        )
    ).lower()

    message = str(
        event.get(
            "message",
            ""
        )
    ).lower()

    if (
        "tamper" in event_type
        or "tamper" in message
    ):
        return "TAMPER"

    if (
        "unknown" in event_type
        or "unknown person" in message
    ):
        return "UNKNOWN"

    if (
        "face" in event_type
        or "recognized" in message
        or "recognised" in message
    ):
        return "FACE"

    if (
        "motion" in event_type
        or "motion" in message
    ):
        return "MOTION"

    if "recording" in event_type:
        return "RECORDING"

    return "SYSTEM"


def event_description(event):
    """
    Produce a human-readable event description.
    """

    if not isinstance(event, dict):
        return safe_text(
            event,
            "System event"
        )

    category = classify_event(
        event
    )

    message = safe_text(
        event.get(
            "message"
        ),
        ""
    )

    person = event.get(
        "person_name"
    )

    confidence = event.get(
        "confidence"
    )

    if message:
        description = message

    elif category == "FACE" and person:
        description = (
            f"Registered person detected: {person}"
        )

    elif category == "UNKNOWN":
        description = (
            "Unknown person detected"
        )

    elif category == "MOTION":
        description = (
            "Motion detected"
        )

    elif category == "TAMPER":
        description = (
            "Camera tamper condition detected"
        )

    else:
        description = safe_text(
            event.get(
                "event_type"
            ),
            "System event"
        )

    if category == "FACE" and person:

        if str(person).lower() not in description.lower():
            description += f" — {person}"

    if category == "FACE" and confidence is not None:

        description += (
            f" | Confidence: "
            f"{confidence_text(confidence)}"
        )

    return description


# ============================================================
# RECORDING FILE STATUS
# ============================================================

def get_recording_path(recording):
    """
    Determine the physical recording file path.
    """

    if not isinstance(recording, dict):
        return None

    filepath = recording.get(
        "filepath"
    )

    if filepath:
        return str(
            filepath
        )

    filename = recording.get(
        "filename"
    )

    if filename:
        return os.path.join(
            BASE_DIR,
            "recordings",
            str(filename)
        )

    return None


def recording_status(recording):
    """
    Determine whether physical video still exists.

    The database row may remain even after the
    physical file is removed by the 15-day cleanup.
    """

    filepath = get_recording_path(
        recording
    )

    if filepath and os.path.exists(filepath):

        return (
            "AVAILABLE",
            GREEN
        )

    return (
        "ARCHIVED — VIDEO FILE REMOVED",
        ORANGE
    )


# ============================================================
# DAILY DATA PREPARATION
# ============================================================

def prepare_daily_data(
    selected_date,
    events,
    recordings,
    tamper_events
):
    """
    Filter and organize data belonging to selected date.
    """

    events = normalize_report_items(
        events,
        "event"
    )

    recordings = normalize_report_items(
        recordings,
        "recording"
    )

    tamper_events = normalize_report_items(
        tamper_events,
        "tamper"
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    daily_events = []

    for event in events:

        timestamp = event.get(
            "timestamp"
        )

        dt = parse_timestamp(
            timestamp
        )

        if (
            dt
            and dt.strftime("%Y-%m-%d")
            == selected_date
        ):
            daily_events.append(
                event
            )

        # If timestamp was unavailable, do not
        # incorrectly assign it to this date.

    daily_events.sort(
        key=lambda item: (
            parse_timestamp(
                item.get(
                    "timestamp"
                )
            )
            or datetime.min
        )
    )

    # --------------------------------------------------------
    # RECORDINGS
    # --------------------------------------------------------

    daily_recordings = []

    compact_date = selected_date.replace(
        "-",
        ""
    )

    for recording in recordings:

        created_at = recording.get(
            "created_at"
        )

        dt = parse_timestamp(
            created_at
        )

        filename = safe_text(
            recording.get(
                "filename"
            ),
            ""
        )

        belongs_to_date = False

        if dt:

            belongs_to_date = (
                dt.strftime("%Y-%m-%d")
                == selected_date
            )

        elif compact_date in filename:

            belongs_to_date = True

        if belongs_to_date:

            daily_recordings.append(
                recording
            )

    daily_recordings.sort(
        key=lambda item: (
            parse_timestamp(
                item.get(
                    "created_at"
                )
            )
            or datetime.min
        )
    )

    # --------------------------------------------------------
    # TAMPER EVENTS
    # --------------------------------------------------------

    daily_tamper = []

    for tamper in tamper_events:

        timestamp = tamper.get(
            "timestamp"
        )

        dt = parse_timestamp(
            timestamp
        )

        if (
            dt
            and dt.strftime("%Y-%m-%d")
            == selected_date
        ):
            daily_tamper.append(
                tamper
            )

    daily_tamper.sort(
        key=lambda item: (
            parse_timestamp(
                item.get(
                    "timestamp"
                )
            )
            or datetime.min
        )
    )

    return (
        daily_events,
        daily_recordings,
        daily_tamper
    )


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_daily_report(
    selected_date,
    events,
    recordings,
    tamper_events=None,
    report_until=None
):
    """
    Generate the SENTRIX daily security report.
    """

    # ========================================================
    # VALIDATE DATE
    # ========================================================

    try:
        selected_datetime = datetime.strptime(
            selected_date,
            "%Y-%m-%d"
        )

    except ValueError:

        raise ValueError(
            "selected_date must be in YYYY-MM-DD format."
        )

    # ========================================================
    # REPORT TIME
    # ========================================================

    if report_until is None:
        report_until = datetime.now()

    # If the selected date is today, report only until now.
    # If an older date is selected, use the supplied
    # report_until or end-of-day logic from the caller.
    if not isinstance(report_until, datetime):
        parsed_report_until = parse_timestamp(
            report_until
        )

        if parsed_report_until:
            report_until = parsed_report_until
        else:
            report_until = datetime.now()

    # ========================================================
    # NORMALIZE INPUT
    # ========================================================

    events = normalize_report_items(
        events,
        "event"
    )

    recordings = normalize_report_items(
        recordings,
        "recording"
    )

    tamper_events = normalize_report_items(
        tamper_events,
        "tamper"
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    (
        daily_events,
        daily_recordings,
        daily_tamper
    ) = prepare_daily_data(
        selected_date,
        events,
        recordings,
        tamper_events
    )

    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

    motion_events = [
        e
        for e in daily_events
        if classify_event(e) == "MOTION"
    ]

    face_events = [
        e
        for e in daily_events
        if classify_event(e) == "FACE"
    ]

    unknown_events = [
        e
        for e in daily_events
        if classify_event(e) == "UNKNOWN"
    ]

    tamper_event_records = [
        e
        for e in daily_events
        if classify_event(e) == "TAMPER"
    ]

    recording_events = [
        e
        for e in daily_events
        if classify_event(e) == "RECORDING"
    ]

    # ========================================================
    # TAMPER BREAKDOWN
    # ========================================================

    camera_moved = 0
    lens_covered = 0
    camera_blurred = 0

    for tamper in daily_tamper:

        tamper_type = str(
            tamper.get(
                "tamper_type",
                ""
            )
        ).lower()

        if "moved" in tamper_type:
            camera_moved += 1

        elif "covered" in tamper_type:
            lens_covered += 1

        elif "blur" in tamper_type:
            camera_blurred += 1

    if camera_moved == 0:

        camera_moved = sum(
            1
            for event in tamper_event_records
            if "moved" in str(
                event.get(
                    "message",
                    ""
                )
            ).lower()
        )

    if lens_covered == 0:

        lens_covered = sum(
            1
            for event in tamper_event_records
            if "covered" in str(
                event.get(
                    "message",
                    ""
                )
            ).lower()
        )

    if camera_blurred == 0:

        camera_blurred = sum(
            1
            for event in tamper_event_records
            if "blur" in str(
                event.get(
                    "message",
                    ""
                )
            ).lower()
        )

    total_tamper = max(
        len(daily_tamper),
        len(tamper_event_records)
    )

    # ========================================================
    # RECORDING ANALYSIS
    # ========================================================

    total_recording_seconds = 0.0

    for recording in daily_recordings:

        try:
            total_recording_seconds += float(
                recording.get(
                    "duration",
                    0
                )
                or 0
            )

        except (
            TypeError,
            ValueError
        ):
            continue

    # ========================================================
    # RECORDING AVAILABILITY
    # ========================================================

    available_recordings = 0
    archived_recordings = 0

    for recording in daily_recordings:

        status, _ = recording_status(
            recording
        )

        if status == "AVAILABLE":
            available_recordings += 1
        else:
            archived_recordings += 1

    # ========================================================
    # PEOPLE DETECTED
    # ========================================================

    recognized_people = []

    for event in face_events:

        person = event.get(
            "person_name"
        )

        if person:

            person = str(
                person
            ).strip()

            if (
                person
                and person not in recognized_people
            ):
                recognized_people.append(
                    person
                )

    # ========================================================
    # SECURITY STATUS
    # ========================================================

    if total_tamper > 0:

        security_status = (
            "ATTENTION REQUIRED"
        )

        security_color = RED

    elif unknown_events:

        security_status = (
            "UNKNOWN ACTIVITY DETECTED"
        )

        security_color = ORANGE

    else:

        security_status = (
            "NORMAL MONITORING"
        )

        security_color = GREEN

    # ========================================================
    # FIRST / LAST EVENT
    # ========================================================

    first_event = (
        daily_events[0]
        if daily_events
        else None
    )

    last_event = (
        daily_events[-1]
        if daily_events
        else None
    )

    # ========================================================
    # REPORT FILE
    # ========================================================

    report_filename = (
        f"SENTRIX_Report_{selected_date}.pdf"
    )

    report_path = os.path.join(
        REPORTS_FOLDER,
        report_filename
    )

    # ========================================================
    # PDF DOCUMENT
    # ========================================================

    document = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm
    )

    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=27,
        leading=31,
        textColor=DARK,
        alignment=TA_LEFT,
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=GREY,
        alignment=TA_LEFT,
        spaceAfter=4
    )

    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=DARK,
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=14,
        textColor=colors.HexColor("#29313D"),
        spaceAfter=6
    )

    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=GREY
    )

    # ========================================================
    # STORY
    # ========================================================

    story = []

    # ========================================================
    # HEADER
    # ========================================================

    header_table = Table(
        [
            [
                Paragraph(
                    "<b>SENTRIX</b>",
                    ParagraphStyle(
                        "HeaderLogo",
                        parent=styles["Normal"],
                        fontName="Helvetica-Bold",
                        fontSize=13,
                        textColor=colors.white
                    )
                ),
                Paragraph(
                    "DAILY SECURITY INTELLIGENCE REPORT",
                    ParagraphStyle(
                        "HeaderRight",
                        parent=styles["Normal"],
                        fontName="Helvetica-Bold",
                        fontSize=8,
                        textColor=colors.white,
                        alignment=TA_CENTER
                    )
                )
            ]
        ],
        colWidths=[
            45 * mm,
            125 * mm
        ]
    )

    header_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                DARK
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        header_table
    )

    story.append(
        Spacer(1, 14)
    )

    story.append(
        Paragraph(
            "Daily Security Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            selected_datetime.strftime(
                "%A, %d %B %Y"
            ),
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Analysis window: 00:00:00 → "
            f"{report_until.strftime('%H:%M:%S')}",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Report generated: "
            f"{report_until.strftime('%d %B %Y, %H:%M:%S')}",
            subtitle_style
        )
    )

    # ========================================================
    # SECURITY STATUS
    # ========================================================

    story.append(
        Spacer(1, 8)
    )

    status_table = Table(
        [
            [
                Paragraph(
                    "<b>CURRENT SECURITY STATE</b>",
                    ParagraphStyle(
                        "StatusLabel",
                        parent=styles["Normal"],
                        fontName="Helvetica-Bold",
                        fontSize=8,
                        textColor=GREY,
                        alignment=TA_CENTER
                    )
                )
            ],
            [
                Paragraph(
                    f"<b>{security_status}</b>",
                    ParagraphStyle(
                        "StatusValue",
                        parent=styles["Normal"],
                        fontName="Helvetica-Bold",
                        fontSize=16,
                        textColor=security_color,
                        alignment=TA_CENTER
                    )
                )
            ]
        ],
        colWidths=[
            170 * mm
        ]
    )

    status_table.setStyle(
        TableStyle([
            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                security_color
            ),
            (
                "LINEBELOW",
                (0, 0),
                (-1, 0),
                0.5,
                BORDER
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(
        status_table
    )

    # ========================================================
    # 01 — EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "01 — What Happened Today",
            section_style
        )
    )

    if not daily_events and not daily_recordings:

        summary_text = (
            f"No security events or motion-triggered "
            f"recordings were recorded for {selected_date} "
            f"within the available monitoring data."
        )

    else:

        summary_parts = []

        summary_parts.append(
            f"SENTRIX recorded {len(daily_events)} "
            f"security event(s) during the analysis window."
        )

        if motion_events:
            summary_parts.append(
                f"{len(motion_events)} motion event(s) "
                f"were detected."
            )

        if face_events:
            summary_parts.append(
                f"{len(face_events)} registered-face "
                f"recognition event(s) were recorded."
            )

        if unknown_events:
            summary_parts.append(
                f"{len(unknown_events)} unknown-person "
                f"event(s) were detected."
            )

        if total_tamper:
            summary_parts.append(
                f"{total_tamper} camera-integrity/tamper "
                f"event(s) were recorded."
            )

        if daily_recordings:
            summary_parts.append(
                f"{len(daily_recordings)} motion-triggered "
                f"video recording(s) were stored, representing "
                f"{format_duration(total_recording_seconds)} "
                f"of recorded footage."
            )

        summary_text = " ".join(
            summary_parts
        )

    story.append(
        Paragraph(
            summary_text,
            body_style
        )
    )

    # ========================================================
    # 02 — DAILY INTELLIGENCE
    # ========================================================

    story.append(
        Paragraph(
            "02 — Daily Intelligence",
            section_style
        )
    )

    intelligence_rows = [
        [
            "FIRST EVENT",
            (
                format_time(
                    first_event.get("timestamp")
                )
                if first_event
                else "No events"
            )
        ],
        [
            "LATEST EVENT",
            (
                format_time(
                    last_event.get("timestamp")
                )
                if last_event
                else "No events"
            )
        ],
        [
            "MOTION ACTIVITY",
            str(len(motion_events))
        ],
        [
            "KNOWN PERSON ACTIVITY",
            str(len(face_events))
        ],
        [
            "UNKNOWN ACTIVITY",
            str(len(unknown_events))
        ],
        [
            "CAMERA INTEGRITY",
            str(total_tamper)
        ],
        [
            "VIDEO EVIDENCE",
            str(len(daily_recordings))
        ],
        [
            "AVAILABLE VIDEOS",
            str(available_recordings)
        ],
        [
            "ARCHIVED VIDEOS",
            str(archived_recordings)
        ],
        [
            "RECORDED DURATION",
            format_duration(
                total_recording_seconds
            )
        ]
    ]

    intelligence_table = Table(
        intelligence_rows,
        colWidths=[
            70 * mm,
            100 * mm
        ]
    )

    intelligence_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                LIGHT_GREY
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8.5
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                DARK
            ),
            (
                "LINEBELOW",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        intelligence_table
    )

    # ========================================================
    # 03 — RECOGNIZED PEOPLE
    # ========================================================

    story.append(
        Paragraph(
            "03 — Recognized People",
            section_style
        )
    )

    if recognized_people:

        people_text = (
            "The following registered identities were detected "
            "during the reporting period: "
            + ", ".join(
                recognized_people
            )
            + "."
        )

    else:

        people_text = (
            "No registered person was identified during the "
            "reporting period."
        )

    story.append(
        Paragraph(
            people_text,
            body_style
        )
    )

    # ========================================================
    # 04 — UNKNOWN ACTIVITY
    # ========================================================

    story.append(
        Paragraph(
            "04 — Unknown-Person Activity",
            section_style
        )
    )

    if unknown_events:

        story.append(
            Paragraph(
                f"SENTRIX recorded {len(unknown_events)} "
                f"unknown-person event(s). These events are "
                f"listed below in chronological order.",
                body_style
            )
        )

        unknown_rows = [
            [
                "TIME",
                "EVENT",
                "DETAIL"
            ]
        ]

        for event in unknown_events:

            unknown_rows.append(
                [
                    format_time(
                        event.get(
                            "timestamp"
                        )
                    ),
                    "UNKNOWN PERSON",
                    event_description(
                        event
                    )
                ]
            )

        unknown_table = Table(
            unknown_rows,
            colWidths=[
                28 * mm,
                38 * mm,
                104 * mm
            ],
            repeatRows=1
        )

        unknown_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    RED
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7.8
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )

        story.append(
            unknown_table
        )

    else:

        story.append(
            Paragraph(
                "No unknown-person events were recorded "
                "during the reporting period.",
                body_style
            )
        )

    # ========================================================
    # 05 — CAMERA INTEGRITY
    # ========================================================

    story.append(
        Paragraph(
            "05 — Camera Integrity",
            section_style
        )
    )

    if daily_tamper:

        integrity_text = (
            "Camera integrity monitoring recorded the following "
            "conditions during the reporting period."
        )

    else:

        integrity_text = (
            "No tamper conditions were recorded during the "
            "reporting period."
        )

    story.append(
        Paragraph(
            integrity_text,
            body_style
        )
    )

    if daily_tamper:

        tamper_rows = [
            [
                "TIME",
                "CONDITION",
                "BRIGHTNESS",
                "BLUR VARIANCE",
                "HIST. CORRELATION"
            ]
        ]

        for tamper in daily_tamper:

            brightness = tamper.get(
                "brightness"
            )

            blur = tamper.get(
                "blur_variance"
            )

            correlation = tamper.get(
                "histogram_correlation"
            )

            try:
                brightness_text = (
                    f"{float(brightness):.2f}"
                    if brightness is not None
                    else "—"
                )
            except (
                TypeError,
                ValueError
            ):
                brightness_text = safe_text(
                    brightness
                )

            try:
                blur_text = (
                    f"{float(blur):.2f}"
                    if blur is not None
                    else "—"
                )
            except (
                TypeError,
                ValueError
            ):
                blur_text = safe_text(
                    blur
                )

            try:
                correlation_text = (
                    f"{float(correlation):.3f}"
                    if correlation is not None
                    else "—"
                )
            except (
                TypeError,
                ValueError
            ):
                correlation_text = safe_text(
                    correlation
                )

            tamper_rows.append(
                [
                    format_time(
                        tamper.get(
                            "timestamp"
                        )
                    ),
                    safe_text(
                        tamper.get(
                            "tamper_type"
                        )
                    ),
                    brightness_text,
                    blur_text,
                    correlation_text
                ]
            )

        tamper_table = Table(
            tamper_rows,
            colWidths=[
                27 * mm,
                43 * mm,
                27 * mm,
                35 * mm,
                38 * mm
            ],
            repeatRows=1
        )

        tamper_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    DARK_BLUE
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER
                ),
                (
                    "ALIGN",
                    (2, 1),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )

        story.append(
            tamper_table
        )

    # ========================================================
    # 06 — VIDEO EVIDENCE
    # ========================================================

    story.append(
        Paragraph(
            "06 — Video Evidence",
            section_style
        )
    )

    if daily_recordings:

        recording_rows = [
            [
                "TIME",
                "FILE",
                "DURATION",
                "STATUS"
            ]
        ]

        for recording in daily_recordings:

            status, status_color = recording_status(
                recording
            )

            recording_rows.append(
                [
                    format_time(
                        recording.get(
                            "created_at"
                        )
                    ),
                    safe_text(
                        recording.get(
                            "filename"
                        )
                    ),
                    format_duration(
                        recording.get(
                            "duration",
                            0
                        )
                    ),
                    Paragraph(
                        f"<b>{status}</b>",
                        ParagraphStyle(
                            "RecordingStatus",
                            parent=styles["Normal"],
                            fontName="Helvetica-Bold",
                            fontSize=6.8,
                            textColor=status_color
                        )
                    )
                ]
            )

        recording_table = Table(
            recording_rows,
            colWidths=[
                27 * mm,
                75 * mm,
                30 * mm,
                38 * mm
            ],
            repeatRows=1
        )

        recording_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    DARK_BLUE
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )

        story.append(
            recording_table
        )

        story.append(
            Spacer(1, 5)
        )

        story.append(
            Paragraph(
                "Video retention policy: SENTRIX keeps physical "
                "recording files for 15 days. After the retention "
                "period, the physical video file is removed while "
                "the database record may remain for historical "
                "reporting. Therefore, an ARCHIVED status means "
                "the event record exists but the physical video "
                "cannot currently be replayed.",
                small_style
            )
        )

    else:

        story.append(
            Paragraph(
                "No video evidence was stored for this reporting "
                "period.",
                body_style
            )
        )

    # ========================================================
    # 07 — COMPLETE TIMELINE
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "07 — Complete Security Timeline",
            section_style
        )
    )

    story.append(
        Paragraph(
            "The following timeline reconstructs the recorded "
            "security activity in chronological order. Every "
            "entry is derived from information supplied to the "
            "report generator.",
            body_style
        )
    )

    if daily_events:

        timeline_rows = [
            [
                "TIME",
                "TYPE",
                "EVENT"
            ]
        ]

        for event in daily_events:

            category = classify_event(
                event
            )

            timeline_rows.append(
                [
                    format_time(
                        event.get(
                            "timestamp"
                        )
                    ),
                    category,
                    event_description(
                        event
                    )
                ]
            )

        timeline_table = Table(
            timeline_rows,
            colWidths=[
                27 * mm,
                27 * mm,
                116 * mm
            ],
            repeatRows=1
        )

        timeline_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    DARK
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7.7
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "ALIGN",
                    (0, 1),
                    (1, -1),
                    "CENTER"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )

        story.append(
            timeline_table
        )

    else:

        story.append(
            Paragraph(
                "No events were recorded for this day.",
                body_style
            )
        )

    # ========================================================
    # 08 — DAY RECONSTRUCTION
    # ========================================================

    story.append(
        Paragraph(
            "08 — Day Reconstruction",
            section_style
        )
    )

    if daily_events:

        narrative_parts = []

        for index, event in enumerate(
            daily_events
        ):

            event_time = format_time(
                event.get(
                    "timestamp"
                )
            )

            description = event_description(
                event
            )

            if index == 0:

                narrative_parts.append(
                    f"At {event_time}, SENTRIX first recorded "
                    f"{description.lower()}."
                )

            else:

                narrative_parts.append(
                    f"At {event_time}, the system recorded "
                    f"{description.lower()}."
                )

        narrative_text = " ".join(
            narrative_parts
        )

        story.append(
            Paragraph(
                narrative_text,
                body_style
            )
        )

    else:

        story.append(
            Paragraph(
                "There is no recorded security activity from "
                "which to construct a day reconstruction.",
                body_style
            )
        )

    # ========================================================
    # 09 — FACE RECOGNITION
    # ========================================================

    if face_events:

        story.append(
            Paragraph(
                "09 — Face Recognition Details",
                section_style
            )
        )

        face_rows = [
            [
                "TIME",
                "PERSON",
                "CONFIDENCE"
            ]
        ]

        for event in face_events:

            face_rows.append(
                [
                    format_time(
                        event.get(
                            "timestamp"
                        )
                    ),
                    safe_text(
                        event.get(
                            "person_name"
                        )
                    ),
                    confidence_text(
                        event.get(
                            "confidence"
                        )
                    )
                ]
            )

        face_table = Table(
            face_rows,
            colWidths=[
                35 * mm,
                80 * mm,
                55 * mm
            ],
            repeatRows=1
        )

        face_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    GREEN
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )

        story.append(
            face_table
        )

    # ========================================================
    # 10 — FINAL SYSTEM ASSESSMENT
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "10 — End-of-Report System Assessment",
            section_style
        )
    )

    if total_tamper > 0:

        final_text = (
            "Camera integrity events were recorded during "
            "the analysis period. The report identifies the "
            "recorded tamper conditions and their associated "
            "sensor values above. These events should be "
            "reviewed against the corresponding video evidence "
            "when that evidence remains available."
        )

    elif unknown_events:

        final_text = (
            "The monitoring period contained unknown-person "
            "activity. The associated timestamps and available "
            "event information are documented in this report. "
            "Corresponding video evidence should be reviewed "
            "when available."
        )

    elif daily_events:

        final_text = (
            "SENTRIX recorded security monitoring activity "
            "during the reporting period. The chronological "
            "event timeline and available video evidence are "
            "included above."
        )

    else:

        final_text = (
            "No security activity was recorded during the "
            "available reporting period."
        )

    story.append(
        Paragraph(
            final_text,
            body_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "<b>Data integrity note:</b> This report describes "
            "only information supplied to the report generator "
            "from the SENTRIX monitoring database or dashboard "
            "data. It does not infer events that were not "
            "recorded.",
            small_style
        )
    )

    # ========================================================
    # FOOTER
    # ========================================================

    def add_footer(canvas, doc):

        canvas.saveState()

        canvas.setStrokeColor(
            BORDER
        )

        canvas.line(
            15 * mm,
            10 * mm,
            PAGE_WIDTH - 15 * mm,
            10 * mm
        )

        canvas.setFont(
            "Helvetica",
            7
        )

        canvas.setFillColor(
            GREY
        )

        canvas.drawString(
            15 * mm,
            6 * mm,
            "SENTRIX • Security Monitoring System"
        )

        canvas.drawRightString(
            PAGE_WIDTH - 15 * mm,
            6 * mm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story,
        onFirstPage=add_footer,
        onLaterPages=add_footer
    )

    return report_path

