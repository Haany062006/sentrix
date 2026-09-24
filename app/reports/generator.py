import os
import re
import math
import tempfile
from datetime import datetime
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
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

REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_FOLDER, exist_ok=True)


# ============================================================
# REPORT SETTINGS
# ============================================================

PAGE_WIDTH, PAGE_HEIGHT = A4

CHART_WIDTH = 175
CHART_HEIGHT = 78

DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
]


# ============================================================
# EVENT PARSING
# ============================================================

def parse_event(event):
    """
    Convert SENTRIX event text into:

        {
            "datetime": datetime,
            "message": str,
            "category": str,
            "person": str | None
        }

    Expected event format:

        2026-09-24 10:15:20 - Motion detected
    """

    if not isinstance(event, str):
        return None

    event = event.strip()

    match = re.match(
        r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*-\s*(.*)$",
        event
    )

    if not match:
        return None

    timestamp_text = match.group(1)
    message = match.group(2).strip()

    timestamp = None

    for fmt in DATE_FORMATS:
        try:
            timestamp = datetime.strptime(timestamp_text, fmt)
            break
        except ValueError:
            continue

    if timestamp is None:
        return None

    message_lower = message.lower()

    # --------------------------------------------------------
    # CATEGORY DETECTION
    # --------------------------------------------------------

    if "tamper" in message_lower:
        category = "Tamper"

    elif "unknown person" in message_lower:
        category = "Unknown Person"

    elif "recognized" in message_lower:
        category = "Known Person"

    elif "motion" in message_lower:
        category = "Motion"

    elif "recording started" in message_lower:
        category = "Recording"

    elif "recording stopped" in message_lower:
        category = "Recording"

    elif "camera" in message_lower:
        category = "Camera"

    else:
        category = "System"

    # --------------------------------------------------------
    # PERSON NAME
    # --------------------------------------------------------

    person = None

    if category == "Known Person":

        person_match = re.search(
            r"(.+?)\s+recognized",
            message,
            re.IGNORECASE
        )

        if person_match:
            person = person_match.group(1).strip()

    return {
        "datetime": timestamp,
        "message": message,
        "category": category,
        "person": person
    }


# ============================================================
# FILTER EVENTS BY DATE
# ============================================================

def prepare_events(selected_date, current_events):

    parsed = []

    for event in current_events or []:

        item = parse_event(event)

        if item is None:
            continue

        if item["datetime"].strftime("%Y-%m-%d") != selected_date:
            continue

        parsed.append(item)

    parsed.sort(
        key=lambda x: x["datetime"]
    )

    return parsed


# ============================================================
# RECORDING PARSER
# ============================================================

def parse_recording_time(filename):

    """
    Tries to extract a timestamp from common SENTRIX
    recording names.

    Example:

        SENTRIX_20260924_101530.mp4
        recording_20260924_101530.mp4
    """

    if not filename:
        return None

    patterns = [
        r"(\d{8})[_-](\d{6})",
        r"(\d{4}-\d{2}-\d{2})[_-](\d{2}-\d{2}-\d{2})",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            filename
        )

        if not match:
            continue

        try:

            if len(match.groups()) == 2:

                first = match.group(1)
                second = match.group(2)

                if len(first) == 8:

                    return datetime.strptime(
                        first + second,
                        "%Y%m%d%H%M%S"
                    )

        except ValueError:
            pass

    return None


# ============================================================
# CHART 1
# ACTIVITY OVER 24 HOURS
# ============================================================

def create_activity_timeline(events, output):

    hourly = [0] * 24

    for event in events:
        hour = event["datetime"].hour
        hourly[hour] += 1

    hours = list(range(24))

    plt.figure(
        figsize=(10, 4.2)
    )

    plt.plot(
        hours,
        hourly,
        marker="o",
        linewidth=2
    )

    plt.xticks(
        hours,
        [f"{h:02d}" for h in hours],
        fontsize=8
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Activity")
    plt.title(
        "SENTRIX Activity Timeline"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CHART 2
# HOURLY CATEGORY ACTIVITY
# ============================================================

def create_hourly_activity(events, output):

    categories = [
        "Motion",
        "Known Person",
        "Unknown Person",
        "Tamper"
    ]

    data = {
        category: [0] * 24
        for category in categories
    }

    for event in events:

        category = event["category"]

        if category not in data:
            continue

        hour = event["datetime"].hour

        data[category][hour] += 1

    hours = np.arange(24)

    plt.figure(
        figsize=(10, 4.5)
    )

    width = 0.18

    for index, category in enumerate(categories):

        positions = (
            hours
            + (index - 1.5) * width
        )

        plt.bar(
            positions,
            data[category],
            width=width,
            label=category
        )

    plt.xticks(
        hours,
        [f"{h:02d}" for h in hours],
        fontsize=8
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Occurrences")

    plt.title(
        "Hourly Security Activity"
    )

    plt.legend(
        fontsize=8,
        ncol=2
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CHART 3
# EVENT DISTRIBUTION
# ============================================================

def create_event_distribution(events, output):

    counter = Counter()

    for event in events:

        category = event["category"]

        if category in [
            "Motion",
            "Known Person",
            "Unknown Person",
            "Tamper",
            "Recording"
        ]:
            counter[category] += 1

    if not counter:
        counter["No Data"] = 1

    labels = list(counter.keys())
    values = list(counter.values())

    plt.figure(
        figsize=(7, 5)
    )

    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Security Activity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CHART 4
# PEOPLE DETECTED
# ============================================================

def create_people_chart(events, output):

    people = Counter()
    unknown = 0

    for event in events:

        if event["category"] == "Known Person":

            name = event["person"]

            if name:
                people[name] += 1

        elif event["category"] == "Unknown Person":

            unknown += 1

    if unknown > 0:
        people["Unknown"] = unknown

    if not people:
        people["No Person Data"] = 0

    labels = list(people.keys())
    values = list(people.values())

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.bar(
        labels,
        values
    )

    plt.title(
        "Human Presence Analysis"
    )

    plt.xlabel("Person")
    plt.ylabel("Detection Activity")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CHART 5
# ACTIVITY HEATMAP
# ============================================================

def create_activity_heatmap(events, output):

    categories = [
        "Motion",
        "Known Person",
        "Unknown Person",
        "Tamper"
    ]

    matrix = np.zeros(
        (len(categories), 24),
        dtype=int
    )

    category_index = {
        category: index
        for index, category in enumerate(categories)
    }

    for event in events:

        category = event["category"]

        if category not in category_index:
            continue

        row = category_index[category]
        column = event["datetime"].hour

        matrix[row][column] += 1

    plt.figure(
        figsize=(11, 4.5)
    )

    plt.imshow(
        matrix,
        aspect="auto"
    )

    plt.xticks(
        range(24),
        [f"{h:02d}" for h in range(24)]
    )

    plt.yticks(
        range(len(categories)),
        categories
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Activity Type")

    plt.title(
        "Security Activity Heatmap"
    )

    plt.colorbar(
        label="Activity Density"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CHART 6
# RECORDING ACTIVITY BY HOUR
# ============================================================

def create_recording_chart(
    selected_date,
    recordings,
    output
):

    hourly = [0] * 24

    for recording in recordings or []:

        if not isinstance(recording, dict):
            continue

        filename = recording.get(
            "name",
            ""
        )

        timestamp = parse_recording_time(
            filename
        )

        if timestamp is None:
            continue

        if timestamp.strftime("%Y-%m-%d") != selected_date:
            continue

        hourly[timestamp.hour] += 1

    plt.figure(
        figsize=(10, 4)
    )

    plt.bar(
        range(24),
        hourly
    )

    plt.xticks(
        range(24),
        [f"{h:02d}" for h in range(24)],
        fontsize=8
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Recording Segments")

    plt.title(
        "Recording Activity by Hour"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# OBSERVATIONS
# ============================================================

def generate_observations(
    events,
    recordings
):

    observations = []

    if not events:

        observations.append(
            "No event data was available for the selected reporting period."
        )

        return observations

    # --------------------------------------------------------
    # PEAK HOUR
    # --------------------------------------------------------

    hourly = Counter(
        event["datetime"].hour
        for event in events
    )

    if hourly:

        peak_hour, peak_count = (
            hourly.most_common(1)[0]
        )

        observations.append(
            f"Highest recorded activity occurred around "
            f"{peak_hour:02d}:00, with {peak_count} "
            f"recorded activity entries."
        )

    # --------------------------------------------------------
    # UNKNOWN PERSON
    # --------------------------------------------------------

    unknown_count = sum(
        1
        for event in events
        if event["category"] == "Unknown Person"
    )

    if unknown_count:

        observations.append(
            f"Unknown-person detections were recorded "
            f"{unknown_count} time(s) during the reporting period."
        )

    # --------------------------------------------------------
    # TAMPER
    # --------------------------------------------------------

    tamper_count = sum(
        1
        for event in events
        if event["category"] == "Tamper"
    )

    if tamper_count:

        observations.append(
            f"Camera tamper activity was recorded "
            f"{tamper_count} time(s)."
        )

    # --------------------------------------------------------
    # KNOWN PEOPLE
    # --------------------------------------------------------

    people = Counter()

    for event in events:

        if (
            event["category"] == "Known Person"
            and event["person"]
        ):
            people[event["person"]] += 1

    if people:

        top_person, count = (
            people.most_common(1)[0]
        )

        observations.append(
            f"The highest-recognized-person activity "
            f"was associated with {top_person}, "
            f"with {count} recognition event(s)."
        )

    # --------------------------------------------------------
    # RECORDINGS
    # --------------------------------------------------------

    if recordings:

        observations.append(
            f"{len(recordings)} recording file(s) were "
            f"available in the recording repository."
        )

    return observations


# ============================================================
# PDF HEADER / FOOTER
# ============================================================

def add_page_number(canvas, doc):

    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.drawString(
        20 * mm,
        10 * mm,
        "SENTRIX — Intelligent Security Monitoring"
    )

    canvas.drawRightString(
        PAGE_WIDTH - 20 * mm,
        10 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ============================================================
# MAIN REPORT GENERATOR
# ============================================================

def generate_daily_report(
    selected_date,
    current_events,
    recordings
):

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    events = prepare_events(
        selected_date,
        current_events
    )

    # --------------------------------------------------------
    # TEMP CHART DIRECTORY
    # --------------------------------------------------------

    chart_dir = tempfile.mkdtemp(
        prefix="sentrix_report_"
    )

    chart_files = {}

    chart_files["timeline"] = os.path.join(
        chart_dir,
        "activity_timeline.png"
    )

    chart_files["hourly"] = os.path.join(
        chart_dir,
        "hourly_activity.png"
    )

    chart_files["distribution"] = os.path.join(
        chart_dir,
        "event_distribution.png"
    )

    chart_files["people"] = os.path.join(
        chart_dir,
        "people_analysis.png"
    )

    chart_files["heatmap"] = os.path.join(
        chart_dir,
        "activity_heatmap.png"
    )

    chart_files["recording"] = os.path.join(
        chart_dir,
        "recording_activity.png"
    )

    # --------------------------------------------------------
    # CREATE CHARTS
    # --------------------------------------------------------

    create_activity_timeline(
        events,
        chart_files["timeline"]
    )

    create_hourly_activity(
        events,
        chart_files["hourly"]
    )

    create_event_distribution(
        events,
        chart_files["distribution"]
    )

    create_people_chart(
        events,
        chart_files["people"]
    )

    create_activity_heatmap(
        events,
        chart_files["heatmap"]
    )

    create_recording_chart(
        selected_date,
        recordings,
        chart_files["recording"]
    )

    # --------------------------------------------------------
    # PDF PATH
    # --------------------------------------------------------

    output_file = os.path.join(
        REPORTS_FOLDER,
        f"SENTRIX_Daily_Report_{selected_date}.pdf"
    )

    # --------------------------------------------------------
    # PDF DOCUMENT
    # --------------------------------------------------------

    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        spaceBefore=4,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        spaceAfter=6
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11
    )

    # --------------------------------------------------------
    # STORY
    # --------------------------------------------------------

    story = []

    # ========================================================
    # COVER / OVERVIEW
    # ========================================================

    story.append(
        Spacer(1, 10 * mm)
    )

    story.append(
        Paragraph(
            "SENTRIX",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Intelligent Security Monitoring Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Reporting Date: <b>{selected_date}</b>",
            subtitle_style
        )
    )

    story.append(
        Spacer(1, 4 * mm)
    )

    story.append(
        Paragraph(
            "SECURITY ACTIVITY OVERVIEW",
            section_style
        )
    )

    story.append(
        Paragraph(
            "This report presents a visual analysis of the "
            "security activity recorded by SENTRIX during "
            "the selected reporting period. Instead of "
            "listing individual events, activity is grouped "
            "by time, event type, human presence and recording "
            "behaviour.",
            body_style
        )
    )

    story.append(
        Spacer(1, 3 * mm)
    )

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    story.append(
        Image(
            chart_files["timeline"],
            width=175 * mm,
            height=74 * mm
        )
    )

    story.append(
        Spacer(1, 3 * mm)
    )

    story.append(
        Paragraph(
            "The timeline shows how detected activity "
            "changed throughout the day and helps identify "
            "periods of increased surveillance activity.",
            small_style
        )
    )

    story.append(
        Spacer(1, 6 * mm)
    )

    # --------------------------------------------------------
    # HOURLY ACTIVITY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "HOURLY SECURITY ACTIVITY",
            section_style
        )
    )

    story.append(
        Image(
            chart_files["hourly"],
            width=175 * mm,
            height=78 * mm
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # BEHAVIOUR ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "BEHAVIOUR & PRESENCE ANALYSIS",
            section_style
        )
    )

    story.append(
        Paragraph(
            "SENTRIX separates general motion activity from "
            "face-recognition and camera-security events. "
            "This makes it easier to understand what type of "
            "activity contributed to the observed surveillance "
            "pattern.",
            body_style
        )
    )

    # --------------------------------------------------------
    # DISTRIBUTION + PEOPLE
    # --------------------------------------------------------

    table = Table(
        [
            [
                Image(
                    chart_files["distribution"],
                    width=85 * mm,
                    height=60 * mm
                ),

                Image(
                    chart_files["people"],
                    width=85 * mm,
                    height=60 * mm
                )
            ]
        ],
        colWidths=[
            88 * mm,
            88 * mm
        ]
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                )
            ]
        )
    )

    story.append(
        table
    )

    story.append(
        Spacer(1, 6 * mm)
    )

    story.append(
        Paragraph(
            "The distribution chart summarizes the relative "
            "share of detected activity categories, while "
            "the human-presence chart shows recognition "
            "activity associated with known and unknown "
            "persons.",
            small_style
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "SECURITY ACTIVITY HEATMAP",
            section_style
        )
    )

    story.append(
        Image(
            chart_files["heatmap"],
            width=175 * mm,
            height=72 * mm
        )
    )

    story.append(
        Paragraph(
            "The heatmap represents the concentration of "
            "different security activities across the 24-hour "
            "period. Darker regions indicate higher activity "
            "density within the plotted data.",
            small_style
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # RECORDING ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "RECORDING & SECURITY ANALYSIS",
            section_style
        )
    )

    story.append(
        Paragraph(
            "Recording activity is shown by hour rather than "
            "listing every individual recording file. This "
            "provides a clearer representation of when the "
            "surveillance system was actively creating "
            "recordings.",
            body_style
        )
    )

    story.append(
        Image(
            chart_files["recording"],
            width=175 * mm,
            height=70 * mm
        )
    )

    story.append(
        Spacer(1, 5 * mm)
    )

    # ========================================================
    # ANALYTICAL OBSERVATIONS
    # ========================================================

    story.append(
        Paragraph(
            "ANALYTICAL OBSERVATIONS",
            section_style
        )
    )

    observations = generate_observations(
        events,
        recordings
    )

    observation_rows = []

    for index, observation in enumerate(
        observations,
        start=1
    ):

        observation_rows.append(
            [
                Paragraph(
                    f"<b>{index}</b>",
                    body_style
                ),
                Paragraph(
                    observation,
                    body_style
                )
            ]
        )

    if observation_rows:

        observation_table = Table(
            observation_rows,
            colWidths=[
                12 * mm,
                163 * mm
            ]
        )

        observation_table.setStyle(
            TableStyle(
                [
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),
                    (
                        "LINEBELOW",
                        (0, 0),
                        (-1, -1),
                        0.3,
                        colors.lightgrey
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        5
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        5
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
                ]
            )
        )

        story.append(
            observation_table
        )

    story.append(
        Spacer(1, 7 * mm)
    )

    # ========================================================
    # REPORT METHODOLOGY
    # ========================================================

    story.append(
        Paragraph(
            "REPORT METHODOLOGY",
            section_style
        )
    )

    story.append(
        Paragraph(
            "The report groups surveillance events into "
            "time-based and activity-based categories. "
            "Motion, face recognition, unknown-person "
            "detection and tamper events are represented "
            "through visual summaries. Individual event "
            "messages are intentionally omitted from the "
            "main report to reduce repetitive information "
            "and improve analytical readability.",
            body_style
        )
    )

    story.append(
        Paragraph(
            "SENTRIX Daily Security Report • "
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            small_style
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    # --------------------------------------------------------
    # CLEAN TEMP FILES
    # --------------------------------------------------------

    try:

        for filename in os.listdir(chart_dir):

            filepath = os.path.join(
                chart_dir,
                filename
            )

            if os.path.isfile(filepath):

                os.remove(filepath)

        os.rmdir(chart_dir)

    except Exception:
        pass

    return output_file