import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


REPORTS_FOLDER = "reports"


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

    # --------------------------------
    # FILTER RECORDINGS BY DATE
    # --------------------------------

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

    # --------------------------------
    # FILTER EVENTS BY DATE
    # --------------------------------

    date_events = []

    for event in events:

        if selected_date in event:

            date_events.append(event)

    # --------------------------------
    # COUNT EVENTS
    # --------------------------------

    motion_count = 0
    face_count = 0
    unknown_count = 0
    tamper_count = 0
    camera_moved = 0
    lens_covered = 0
    camera_blurred = 0

    for event in date_events:

        text = event.lower()

        if "motion" in text:
            motion_count += 1

        if "face detected" in text:
            face_count += 1

        if "unknown" in text:
            unknown_count += 1

        if "tamper" in text:
            tamper_count += 1

        if "camera moved" in text:
            camera_moved += 1

        if "lens covered" in text:
            lens_covered += 1

        if "camera blurred" in text:
            camera_blurred += 1

    # --------------------------------
    # PDF DOCUMENT
    # --------------------------------

    document = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]

    story = []

    # --------------------------------
    # TITLE
    # --------------------------------

    story.append(
        Paragraph(
            "SENTRIX",
            title_style
        )
    )

    story.append(
        Paragraph(
            "DAILY SECURITY REPORT",
            heading_style
        )
    )

    story.append(
        Spacer(1, 8)
    )

    generated_time = datetime.now().strftime(
        "%d-%m-%Y %I:%M:%S %p"
    )

    story.append(
        Paragraph(
            f"<b>Report Date:</b> {selected_date}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> {generated_time}",
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------
    # SECURITY SUMMARY
    # --------------------------------

    story.append(
        Paragraph(
            "Security Summary",
            heading_style
        )
    )

    summary_data = [
        ["Metric", "Count"],
        ["Motion Events", str(motion_count)],
        ["Recordings", str(len(date_recordings))],
        ["Faces Detected", str(face_count)],
        ["Unknown Faces", str(unknown_count)],
        ["Tamper Incidents", str(tamper_count)]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            90 * mm,
            50 * mm
        ]
    )

    summary_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#222222")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------
    # TAMPER SUMMARY
    # --------------------------------

    story.append(
        Paragraph(
            "Tamper Detection",
            heading_style
        )
    )

    tamper_data = [
        ["Incident", "Count"],
        ["Camera Moved", str(camera_moved)],
        ["Lens Covered", str(lens_covered)],
        ["Camera Blurred", str(camera_blurred)]
    ]

    tamper_table = Table(
        tamper_data,
        colWidths=[
            90 * mm,
            50 * mm
        ]
    )

    tamper_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#222222")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        tamper_table
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------
    # RECORDINGS
    # --------------------------------

    story.append(
        Paragraph(
            "Recordings",
            heading_style
        )
    )

    if date_recordings:

        for recording in date_recordings:

            story.append(
                Paragraph(
                    recording,
                    normal_style
                )
            )

            story.append(
                Spacer(1, 3)
            )

    else:

        story.append(
            Paragraph(
                "No recordings found for this date.",
                normal_style
            )
        )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------
    # EVENT LOG
    # --------------------------------

    story.append(
        Paragraph(
            "Event Log",
            heading_style
        )
    )

    if date_events:

        for event in date_events:

            story.append(
                Paragraph(
                    event,
                    normal_style
                )
            )

            story.append(
                Spacer(1, 4)
            )

    else:

        story.append(
            Paragraph(
                "No events recorded for this date.",
                normal_style
            )
        )

    # --------------------------------
    # BUILD PDF
    # --------------------------------

    document.build(story)

    return filepath