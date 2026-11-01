import os
import tempfile
import shutil
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from app.reports.visualizer import report_visualizer
from app.database.db import (
    get_events_by_date,
    get_hourly_activity,
    get_tamper_summary,
    get_event_type_distribution
)


# ============================================================
# PATHS
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


# ============================================================
# COLORS
# ============================================================

NAVY = colors.HexColor("#0F172A")
DARK_BLUE = colors.HexColor("#1E3A8A")
BLUE = colors.HexColor("#2563EB")
LIGHT_BLUE = colors.HexColor("#EFF6FF")

GREEN = colors.HexColor("#16A34A")
LIGHT_GREEN = colors.HexColor("#F0FDF4")

RED = colors.HexColor("#DC2626")
LIGHT_RED = colors.HexColor("#FEF2F2")

ORANGE = colors.HexColor("#EA580C")
LIGHT_ORANGE = colors.HexColor("#FFF7ED")

WHITE = colors.white

TEXT = colors.HexColor("#1E293B")
SECONDARY = colors.HexColor("#64748B")

GRAY_50 = colors.HexColor("#F8FAFC")
GRAY_100 = colors.HexColor("#F1F5F9")
GRAY_200 = colors.HexColor("#E2E8F0")
GRAY_300 = colors.HexColor("#CBD5E1")


# ============================================================
# PAGE HEADER / FOOTER
# ============================================================

def draw_page(canvas, document):

    canvas.saveState()

    width, height = A4

    # Top bar
    canvas.setFillColor(NAVY)

    canvas.rect(
        0,
        height - 14 * mm,
        width,
        14 * mm,
        fill=1,
        stroke=0
    )

    canvas.setFillColor(WHITE)

    canvas.setFont(
        "Helvetica-Bold",
        10
    )

    canvas.drawString(
        18 * mm,
        height - 8.5 * mm,
        "SENTRIX"
    )

    canvas.setFont(
        "Helvetica",
        7.5
    )

    canvas.drawRightString(
        width - 18 * mm,
        height - 8.5 * mm,
        "AIoT SMART SURVEILLANCE SYSTEM"
    )

    # Footer line
    canvas.setStrokeColor(GRAY_300)
    canvas.setLineWidth(0.5)

    canvas.line(
        18 * mm,
        12 * mm,
        width - 18 * mm,
        12 * mm
    )

    # Footer text
    canvas.setFillColor(SECONDARY)

    canvas.setFont(
        "Helvetica",
        7
    )

    canvas.drawString(
        18 * mm,
        7 * mm,
        "SENTRIX • Automated Security Intelligence"
    )

    canvas.drawRightString(
        width - 18 * mm,
        7 * mm,
        f"Page {document.page}"
    )

    canvas.restoreState()


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_daily_report(
    selected_date,
    events,
    recordings
):

    os.makedirs(
        REPORTS_FOLDER,
        exist_ok=True
    )

    filename = (
        f"SENTRIX_Report_{selected_date}.pdf"
    )

    filepath = os.path.join(
        REPORTS_FOLDER,
        filename
    )

    # ========================================================
    # FILTER RECORDINGS
    # ========================================================

    date_recordings = []

    for recording in recordings:

        name = recording.get(
            "name",
            ""
        )

        if selected_date in name:

            date_recordings.append(
                name
            )

    # ========================================================
    # FILTER EVENTS (IN-MEMORY + FULL DATABASE HISTORY)
    # ========================================================

    date_events = []

    for event in events:
        event_text = str(event)
        if selected_date in event_text:
            date_events.append(event_text)

    # Enrich with all events recorded in SQLite for selected_date
    try:
        db_events = get_events_by_date(selected_date)
        existing_set = set(date_events)
        for dbe in db_events:
            ts = dbe.get("timestamp", "").replace("T", " ")
            msg = dbe.get("message", "")
            formatted = f"{ts} - {msg}"
            if formatted not in existing_set:
                date_events.append(formatted)
                existing_set.add(formatted)
    except Exception as exc:
        print("Report generator database sync error:", exc)

    # Sort chronologically
    date_events.sort()

    # ========================================================
    # COUNTERS
    # ========================================================

    motion_count = 0
    face_count = 0
    unknown_count = 0
    tamper_count = 0

    camera_moved = 0
    lens_covered = 0
    camera_blurred = 0

    siren_count = 0
    object_missing_count = 0
    object_missing_records = []

    recognized_people = {}

    # ========================================================
    # ANALYZE EVENTS
    # ========================================================

    for event in date_events:

        text = event.lower()

        # Motion
        if "motion detected" in text:
            motion_count += 1

        # Known face
        if "recognized" in text:

            face_count += 1

            try:

                if " - " in event:
                    message = event.split(
                        " - ",
                        1
                    )[1]
                else:
                    message = event

                person = message.split(
                    " recognized",
                    1
                )[0].strip()

                if person:
                    recognized_people[person] = (
                        recognized_people.get(
                            person,
                            0
                        ) + 1
                    )

            except Exception:
                pass

        # Unknown
        if "unknown person" in text:
            unknown_count += 1

        # Tamper
        if "tamper detected" in text:
            tamper_count += 1

        if "camera moved" in text:
            camera_moved += 1

        if "lens covered" in text:
            lens_covered += 1

        if "camera blurred" in text:
            camera_blurred += 1

        # Siren Deterrent Alarm
        if "siren" in text:
            siren_count += 1

        # Monitored Object Missing Alerts
        if "monitored object" in text or "missing alert" in text or "object_alert" in text:
            object_missing_count += 1
            ts = event.split(" - ")[0].strip() if " - " in event else ""
            msg = event.split(" - ", 1)[1].strip() if " - " in event else event
            object_missing_records.append({
                "timestamp": ts,
                "message": msg
            })

    # Fetch structured missing intervals from database
    missing_intervals = []
    try:
        from app.database.db import get_missing_intervals
        missing_intervals = get_missing_intervals(report_date)
    except Exception:
        missing_intervals = []

    if missing_intervals:
        object_missing_count = len(missing_intervals)

    # ========================================================
    # SECURITY STATUS
    # ========================================================

    if tamper_count > 0:

        security_status = "ATTENTION REQUIRED"
        status_color = RED
        status_background = LIGHT_RED

    elif unknown_count > 0:

        security_status = "UNKNOWN ACTIVITY"
        status_color = ORANGE
        status_background = LIGHT_ORANGE

    else:

        security_status = "SYSTEM SECURE"
        status_color = GREEN
        status_background = LIGHT_GREEN

    # ========================================================
    # DOCUMENT
    # ========================================================

    document = SimpleDocTemplate(
        filepath,
        pagesize=A4,

        rightMargin=18 * mm,
        leftMargin=18 * mm,

        topMargin=21 * mm,
        bottomMargin=18 * mm,

        title=f"SENTRIX Daily Security Report {selected_date}",

        author="SENTRIX AIoT Smart Surveillance System"
    )

    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SENTRIXTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=30,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        "SENTRIXSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=SECONDARY
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=NAVY,
        spaceBefore=5,
        spaceAfter=7
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.5,
        textColor=TEXT
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=SECONDARY
    )

    center_style = ParagraphStyle(
        "Center",
        parent=normal_style,
        alignment=TA_CENTER
    )

    white_left_style = ParagraphStyle(
        "WhiteLeft",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=WHITE,
        alignment=TA_LEFT
    )

    white_center_style = ParagraphStyle(
        "WhiteCenter",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=WHITE,
        alignment=TA_CENTER
    )

    metric_number_style = ParagraphStyle(
        "MetricNumber",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=22,
        textColor=NAVY,
        alignment=TA_CENTER
    )

    metric_label_style = ParagraphStyle(
        "MetricLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=SECONDARY,
        alignment=TA_CENTER
    )

    status_title_style = ParagraphStyle(
        "StatusTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=SECONDARY
    )

    status_value_style = ParagraphStyle(
        "StatusValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=status_color,
        alignment=TA_CENTER
    )

    # ========================================================
    # STORY
    # ========================================================

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    story.append(
        Paragraph(
            "SENTRIX",
            title_style
        )
    )

    story.append(
        Paragraph(
            "DAILY SECURITY INTELLIGENCE REPORT",
            subtitle_style
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    # ========================================================
    # REPORT INFORMATION
    # ========================================================

    generated_time = datetime.now().strftime(
        "%d %B %Y • %I:%M:%S %p"
    )

    info_data = [
        [
            Paragraph(
                "<b>REPORT DATE</b><br/>"
                f"{selected_date}",
                normal_style
            ),

            Paragraph(
                "<b>GENERATED</b><br/>"
                f"{generated_time}",
                normal_style
            ),

            Paragraph(
                "<b>PLATFORM</b><br/>"
                "SENTRIX AIoT",
                normal_style
            )
        ]
    ]

    info_table = Table(
        info_data,
        colWidths=[
            55 * mm,
            65 * mm,
            55 * mm
        ]
    )

    info_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                GRAY_50
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                GRAY_300
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                GRAY_200
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
                9
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(info_table)

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    # ========================================================
    # SECURITY STATUS
    # ========================================================

    status_table = Table(
        [
            [
                Paragraph(
                    "CURRENT SECURITY STATUS",
                    status_title_style
                ),
                Paragraph(
                    security_status,
                    status_value_style
                )
            ]
        ],
        colWidths=[
            90 * mm,
            85 * mm
        ]
    )

    status_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                status_background
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                status_color
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

    story.append(status_table)

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ========================================================
    # SECURITY OVERVIEW
    # ========================================================

    story.append(
        Paragraph(
            "Security Overview",
            section_style
        )
    )

    metrics = [
        (motion_count, "MOTION EVENTS"),
        (len(date_recordings), "RECORDINGS"),
        (face_count, "KNOWN FACES"),
        (unknown_count, "UNKNOWN"),
        (tamper_count, "TAMPER")
    ]

    metric_cells = []

    for value, label in metrics:

        metric_cells.append(
            [
                Paragraph(
                    str(value),
                    metric_number_style
                ),
                Paragraph(
                    label,
                    metric_label_style
                )
            ]
        )

    metric_table = Table(
        [metric_cells],
        colWidths=[
            35 * mm,
            35 * mm,
            35 * mm,
            35 * mm,
            35 * mm
        ],
        rowHeights=[
            27 * mm
        ]
    )

    metric_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                WHITE
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                GRAY_300
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                GRAY_200
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(metric_table)

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ========================================================
    # VISUAL SECURITY ANALYTICS (EXAMINER RECOMMENDATION)
    # ========================================================

    story.append(
        Paragraph(
            "Visual Security Analytics",
            section_style
        )
    )

    temp_chart_dir = tempfile.mkdtemp(prefix="sentrix_charts_")
    try:
        donut_chart_path = os.path.join(temp_chart_dir, "event_dist.png")
        incidents_chart_path = os.path.join(temp_chart_dir, "incidents.png")

        # Distribution dictionary
        dist_dict = {
            "motion": motion_count,
            "face_recognized": face_count,
            "unknown_person": unknown_count,
            "tamper": tamper_count,
            "siren_alarm": siren_count,
            "object_missing": object_missing_count
        }

        # Incidents dictionary
        incidents_dict = {
            "CAMERA MOVED": camera_moved,
            "LENS COVERED": lens_covered,
            "CAMERA BLURRED": camera_blurred,
            "SIREN ALARM": siren_count,
            "OBJECT MISSING": object_missing_count
        }

        report_visualizer.generate_event_distribution_chart(dist_dict, output_path=donut_chart_path)
        report_visualizer.generate_security_incidents_chart(incidents_dict, output_path=incidents_chart_path)

        # Place the two side-by-side charts
        chart_table = Table(
            [
                [
                    Image(donut_chart_path, width=86 * mm, height=58 * mm),
                    Image(incidents_chart_path, width=86 * mm, height=58 * mm)
                ]
            ],
            colWidths=[87.5 * mm, 87.5 * mm]
        )
        chart_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(chart_table)

    except Exception as chart_err:
        print("Visual analytics chart generation error:", chart_err)

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    story.append(PageBreak())

    # ========================================================
    # FACE RECOGNITION
    # ========================================================

    story.append(
        Paragraph(
            "Face Recognition Analysis",
            section_style
        )
    )

    people_data = [
        [
            Paragraph(
                "PERSON",
                white_left_style
            ),
            Paragraph(
                "DETECTIONS",
                white_center_style
            ),
            Paragraph(
                "STATUS",
                white_center_style
            )
        ]
    ]

    for person, count in sorted(
        recognized_people.items()
    ):

        people_data.append(
            [
                Paragraph(
                    person,
                    normal_style
                ),
                Paragraph(
                    str(count),
                    center_style
                ),
                Paragraph(
                    "RECOGNIZED",
                    ParagraphStyle(
                        "RecognizedStatus",
                        parent=small_style,
                        fontName="Helvetica-Bold",
                        textColor=GREEN,
                        alignment=TA_CENTER
                    )
                )
            ]
        )

    if unknown_count > 0:

        people_data.append(
            [
                Paragraph(
                    "Unknown Person",
                    normal_style
                ),
                Paragraph(
                    str(unknown_count),
                    center_style
                ),
                Paragraph(
                    "ALERT",
                    ParagraphStyle(
                        "UnknownStatus",
                        parent=small_style,
                        fontName="Helvetica-Bold",
                        textColor=RED,
                        alignment=TA_CENTER
                    )
                )
            ]
        )

    if len(people_data) == 1:

        people_data.append(
            [
                Paragraph(
                    "No persons recognized",
                    small_style
                ),
                Paragraph(
                    "0",
                    center_style
                ),
                Paragraph(
                    "NO DATA",
                    small_style
                )
            ]
        )

    people_table = Table(
        people_data,
        colWidths=[
            90 * mm,
            35 * mm,
            50 * mm
        ],
        repeatRows=1
    )

    people_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                NAVY
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                WHITE
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                GRAY_300
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    WHITE,
                    GRAY_50
                ]
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
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(people_table)

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ========================================================
    # TAMPER ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "Camera Integrity & Tamper Analysis",
            section_style
        )
    )

    tamper_data = [
        [
            Paragraph(
                "INCIDENT TYPE",
                white_left_style
            ),
            Paragraph(
                "OCCURRENCES",
                white_center_style
            ),
            Paragraph(
                "STATUS",
                white_center_style
            )
        ],
        [
            Paragraph(
                "Camera Moved",
                normal_style
            ),
            Paragraph(
                str(camera_moved),
                center_style
            ),
            Paragraph(
                "ALERT" if camera_moved else "CLEAR",
                ParagraphStyle(
                    "MovedStatus",
                    parent=small_style,
                    fontName="Helvetica-Bold",
                    textColor=RED if camera_moved else GREEN,
                    alignment=TA_CENTER
                )
            )
        ],
        [
            Paragraph(
                "Lens Covered",
                normal_style
            ),
            Paragraph(
                str(lens_covered),
                center_style
            ),
            Paragraph(
                "ALERT" if lens_covered else "CLEAR",
                ParagraphStyle(
                    "CoveredStatus",
                    parent=small_style,
                    fontName="Helvetica-Bold",
                    textColor=RED if lens_covered else GREEN,
                    alignment=TA_CENTER
                )
            )
        ],
        [
            Paragraph(
                "Camera Blurred",
                normal_style
            ),
            Paragraph(
                str(camera_blurred),
                center_style
            ),
            Paragraph(
                "ALERT" if camera_blurred else "CLEAR",
                ParagraphStyle(
                    "BlurStatus",
                    parent=small_style,
                    fontName="Helvetica-Bold",
                    textColor=RED if camera_blurred else GREEN,
                    alignment=TA_CENTER
                )
            )
        ]
    ]

    tamper_table = Table(
        tamper_data,
        colWidths=[
            85 * mm,
            40 * mm,
            50 * mm
        ],
        repeatRows=1
    )

    tamper_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                NAVY
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                WHITE
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                GRAY_300
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    WHITE,
                    GRAY_50
                ]
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
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(tamper_table)

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ========================================================
    # MONITORED ASSET & MISSING OBJECT TIMELINE
    # ========================================================

    story.append(
        Paragraph(
            "Monitored Asset & Missing Object Timeline",
            section_style
        )
    )

    story.append(
        Paragraph(
            "Detailed log indicating exact timestamps when the monitored object was absent or missing from camera view.",
            ParagraphStyle("SubTextObj", parent=styles["Normal"], fontSize=8.5, textColor=SECONDARY, spaceAfter=8)
        )
    )

    if missing_intervals:
        obj_table_data = [
            [
                Paragraph("#", white_center_style),
                Paragraph("MONITORED ASSET", white_left_style),
                Paragraph("TIME PERIOD (FROM - TO)", white_left_style),
                Paragraph("DURATION MISSING", white_center_style),
                Paragraph("STATUS", white_center_style)
            ]
        ]

        for idx, inter in enumerate(missing_intervals[:30], start=1):
            asset_title = inter.get("asset_name") or "Monitored Asset"
            start_t = inter.get("start_time") or "N/A"
            end_t = inter.get("end_time")
            dur_str = inter.get("duration_str") or (f"{inter.get('duration_seconds', 0)}s" if inter.get("duration_seconds") else "Ongoing")
            status = inter.get("status", "CURRENTLY MISSING")

            if end_t:
                time_range = f"<b>{start_t}</b><br/>to <b>{end_t}</b>"
            else:
                time_range = f"<b>{start_t}</b><br/><font color='#DC2626'>to Ongoing</font>"

            if status == "RESTORED":
                status_html = "<font color='#059669'><b>RESTORED</b></font>"
            else:
                status_html = "<font color='#DC2626'><b>CURRENTLY MISSING</b></font>"

            obj_table_data.append([
                Paragraph(str(idx), center_style),
                Paragraph(f"<b>{asset_title}</b>", small_style),
                Paragraph(time_range, small_style),
                Paragraph(dur_str, center_style),
                Paragraph(status_html, center_style)
            ])

        obj_table = Table(
            obj_table_data,
            colWidths=[12 * mm, 40 * mm, 62 * mm, 33 * mm, 28 * mm]
        )
        obj_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (4, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, GRAY_300),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_50]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(obj_table)
    elif object_missing_records:
        obj_table_data = [
            [
                Paragraph("#", white_center_style),
                Paragraph("TIMESTAMP (WHEN MISSING)", white_left_style),
                Paragraph("MONITORED ASSET", white_left_style),
                Paragraph("STATUS", white_center_style),
                Paragraph("INCIDENT DETAILS", white_left_style)
            ]
        ]

        for idx, rec in enumerate(object_missing_records[:25], start=1):
            ts_text = rec.get("timestamp", "N/A")
            obj_table_data.append([
                Paragraph(str(idx), center_style),
                Paragraph(f"<b>{ts_text}</b>", small_style),
                Paragraph("Monitored Asset", small_style),
                Paragraph("<font color='#DC2626'><b>MISSING ALERT</b></font>", center_style),
                Paragraph("Asset absent / not detected in camera view", small_style)
            ])

        obj_table = Table(
            obj_table_data,
            colWidths=[12 * mm, 45 * mm, 35 * mm, 35 * mm, 48 * mm]
        )
        obj_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (3, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, GRAY_300),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY_50]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(obj_table)
    else:
        clean_box = Table(
            [[
                Paragraph(
                    "<b>Monitored Asset Integrity:</b> All monitored assets remained securely in camera view throughout surveillance with <b>0 missing incidents</b> on this date.",
                    ParagraphStyle("CleanMsg", parent=styles["Normal"], fontSize=9.5, textColor=TEXT)
                )
            ]],
            colWidths=[175 * mm]
        )
        clean_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREEN),
            ("BOX", (0, 0), (-1, -1), 1, GREEN),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(clean_box)

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ========================================================
    # VIDEO EVIDENCE
    # ========================================================

    story.append(
        Paragraph(
            "Video Evidence",
            section_style
        )
    )

    if date_recordings:

        recording_data = [
            [
                Paragraph(
                    "#",
                    white_center_style
                ),
                Paragraph(
                    "RECORDING FILE",
                    white_left_style
                )
            ]
        ]

        for index, recording in enumerate(
            date_recordings,
            start=1
        ):

            recording_data.append(
                [
                    Paragraph(
                        str(index),
                        center_style
                    ),
                    Paragraph(
                        recording,
                        normal_style
                    )
                ]
            )

        recording_table = Table(
            recording_data,
            colWidths=[
                15 * mm,
                160 * mm
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
                    WHITE
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    GRAY_300
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        WHITE,
                        GRAY_50
                    ]
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

        story.append(recording_table)

    else:

        story.append(
            Paragraph(
                "No video recordings were generated for this date.",
                small_style
            )
        )

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ========================================================
    # REPORT NOTE
    # ========================================================

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    note_table = Table(
        [
            [
                Paragraph(
                    "<b>SENTRIX REPORT NOTE</b><br/>"
                    "This report was automatically generated by "
                    "the SENTRIX AIoT Smart Surveillance System. "
                    "It summarizes motion detection, facial "
                    "recognition, video recording and camera "
                    "integrity events captured during the "
                    "selected date.",
                    small_style
                )
            ]
        ],
        colWidths=[
            175 * mm
        ]
    )

    note_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                LIGHT_BLUE
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                BLUE
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

    story.append(note_table)

    # ========================================================
    # BUILD PDF
    # ========================================================

    try:
        document.build(
            story,
            onFirstPage=draw_page,
            onLaterPages=draw_page
        )
    finally:
        try:
            if 'temp_chart_dir' in locals() and os.path.isdir(temp_chart_dir):
                shutil.rmtree(temp_chart_dir, ignore_errors=True)
        except Exception:
            pass

    return filepath