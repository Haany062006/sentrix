import os
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
    # FILTER EVENTS
    # ========================================================

    date_events = []

    for event in events:

        event_text = str(event)

        if selected_date in event_text:

            date_events.append(
                event_text
            )

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
    # EVENT TIMELINE
    # ========================================================

    story.append(
        Paragraph(
            "Security Event Timeline",
            section_style
        )
    )

    if date_events:

        event_data = [
            [
                Paragraph(
                    "TIME",
                    white_left_style
                ),
                Paragraph(
                    "SECURITY EVENT",
                    white_left_style
                )
            ]
        ]

        for event in date_events:

            event_time = ""

            try:
                event_time = event[:19]
            except Exception:
                pass

            event_message = event

            if " - " in event:

                event_message = event.split(
                    " - ",
                    1
                )[1]

            event_lower = event_message.lower()

            if (
                "unknown" in event_lower
                or "tamper" in event_lower
                or "camera moved" in event_lower
                or "lens covered" in event_lower
                or "camera blurred" in event_lower
            ):

                event_color = RED

            elif "recognized" in event_lower:

                event_color = GREEN

            elif "motion" in event_lower:

                event_color = BLUE

            else:

                event_color = TEXT

            event_style = ParagraphStyle(
                "EventStyle",
                parent=normal_style,
                fontSize=8.3,
                leading=11,
                textColor=event_color
            )

            event_data.append(
                [
                    Paragraph(
                        event_time,
                        small_style
                    ),
                    Paragraph(
                        event_message,
                        event_style
                    )
                ]
            )

        event_table = Table(
            event_data,
            colWidths=[
                40 * mm,
                135 * mm
            ],
            repeatRows=1
        )

        event_table.setStyle(
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
                    "TOP"
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

        story.append(event_table)

    else:

        story.append(
            Paragraph(
                "No security events were recorded for this date.",
                small_style
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

    document.build(
        story,
        onFirstPage=draw_page,
        onLaterPages=draw_page
    )

    return filepath