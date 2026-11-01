"""
SENTRIX Security Report Visualizer
Generates high-resolution visualization charts for PDF reports and web dashboard analytics.
"""

import os
import io
import base64
import tempfile
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


class ReportVisualizer:
    def __init__(self):
        # Configure style
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "figure.autolayout": True
        })

    def generate_event_distribution_chart(self, distribution_dict, output_path=None):
        """
        Generates a professional donut chart of event distributions.
        Returns file path or in-memory BytesIO buffer.
        """
        if not distribution_dict or sum(distribution_dict.values()) == 0:
            distribution_dict = {"Normal Activity": 1}

        labels = []
        sizes = []
        palette = {
            "motion": "#2563EB",            # Blue
            "face_recognized": "#16A34A",   # Green
            "unknown_person": "#EA580C",    # Orange
            "tamper": "#DC2626",            # Red
            "recording_start": "#64748B",   # Slate
            "recording_stop": "#94A3B8",    # Light slate
            "siren_alarm": "#DC2626",      # Red (Siren)
            "object_missing": "#EA580C",   # Orange (Object Missing)
            "system": "#475569"             # Dark slate
        }

        friendly_names = {
            "motion": "Motion Detected",
            "face_recognized": "Known Personnel",
            "unknown_person": "Unknown Person Alert",
            "tamper": "Camera Tamper Alert",
            "recording_start": "Recordings",
            "siren_alarm": "Siren Alarm Trigger",
            "object_missing": "Object Missing Alert"
        }

        colors = []
        for k, v in distribution_dict.items():
            if k in ["recording_stop", "system"]:
                continue
            if v > 0:
                labels.append(friendly_names.get(k, k.replace("_", " ").title()))
                sizes.append(v)
                colors.append(palette.get(k, "#3B82F6"))

        if not sizes:
            labels = ["Standard Surveillance"]
            sizes = [1]
            colors = ["#2563EB"]

        fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=140,
            colors=colors,
            wedgeprops=dict(width=0.42, edgecolor="#FFFFFF", linewidth=1.5)
        )

        for autotext in autotexts:
            autotext.set_color("#FFFFFF")
            autotext.set_fontsize(8)
            autotext.set_weight("bold")

        ax.legend(
            wedges,
            [f"{l} ({s})" for l, s in zip(labels, sizes)],
            title="Event Category",
            loc="center left",
            bbox_to_anchor=(0.95, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9
        )

        total_count = sum(sizes)
        ax.text(
            0, 0,
            f"TOTAL\n{total_count}\nEVENTS",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="#0F172A"
        )

        ax.set_title("Security Event Distribution", pad=12, color="#0F172A", weight="bold")

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=200, facecolor="#FFFFFF")
            plt.close(fig)
            return output_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="#FFFFFF")
            plt.close(fig)
            buf.seek(0)
            return buf

    def generate_hourly_timeline_chart(self, hourly_counts, output_path=None):
        """
        Generates a 24-hour bar chart displaying detection frequency across hours.
        """
        if not hourly_counts or len(hourly_counts) != 24:
            hourly_counts = [0] * 24

        hours = [f"{h:02d}h" for h in range(24)]
        counts = list(hourly_counts)

        fig, ax = plt.subplots(figsize=(6.5, 2.6), dpi=200)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8FAFC")

        bar_colors = ["#2563EB" if c < max(counts or [1])*0.75 else "#EA580C" for c in counts]
        bars = ax.bar(hours, counts, color=bar_colors, width=0.68, edgecolor="#FFFFFF", linewidth=0.5)

        ax.set_title("24-Hour Surveillance Activity Timeline", pad=10, color="#0F172A", weight="bold")
        ax.set_xlabel("Hour of Day (00:00 - 23:00)", color="#64748B", labelpad=6)
        ax.set_ylabel("Security Events", color="#64748B", labelpad=6)

        # Subtle grid
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#CBD5E1")
        ax.set_axisbelow(True)

        # Style ticks and spines
        ax.tick_params(axis="x", rotation=45, labelsize=7.5, colors="#475569")
        ax.tick_params(axis="y", labelsize=8, colors="#475569")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#CBD5E1")

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=200, facecolor="#FFFFFF")
            plt.close(fig)
            return output_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="#FFFFFF")
            plt.close(fig)
            buf.seek(0)
            return buf

    def generate_security_incidents_chart(self, incidents_dict, output_path=None):
        """
        Generates a horizontal bar chart of security incidents, tamper events, and siren alarms.
        """
        labels = [
            "Camera Moved",
            "Lens Covered",
            "Camera Blurred",
            "Siren Alarm",
            "Object Missing"
        ]
        keys = [
            "CAMERA MOVED",
            "LENS COVERED",
            "CAMERA BLURRED",
            "SIREN ALARM",
            "OBJECT MISSING"
        ]
        values = [incidents_dict.get(k, 0) for k in keys]

        fig, ax = plt.subplots(figsize=(5.5, 2.5), dpi=200)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8FAFC")

        bar_colors = ["#DC2626" if v > 0 else "#94A3B8" for v in values]
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=bar_colors, height=0.55, edgecolor="#FFFFFF")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5, color="#1E293B", weight="bold")
        ax.invert_yaxis()  # top-down

        # Value annotations
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.15,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                ha="left",
                fontsize=8,
                weight="bold",
                color="#0F172A"
            )

        max_val = max(values or [1])
        ax.set_xlim(0, max(max_val + 2, 5))
        ax.set_title("Tamper Incidents & Siren Deterrent", pad=10, color="#0F172A", weight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3, color="#CBD5E1")
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#CBD5E1")

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=200, facecolor="#FFFFFF")
            plt.close(fig)
            return output_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="#FFFFFF")
            plt.close(fig)
            buf.seek(0)
            return buf

    def get_dashboard_charts_base64(self, date_str=None):
        """
        Returns base64 encoded strings of all charts for the web dashboard.
        """
        from app.database.db import (
            get_event_type_distribution,
            get_hourly_activity,
            get_tamper_summary
        )

        dist = get_event_type_distribution(date_str)
        hourly = get_hourly_activity(date_str or "")
        tamper = get_tamper_summary(date_str)

        # Merge siren trigger count into incidents
        incidents = dict(tamper)
        incidents["SIREN ALARM"] = dist.get("siren_trigger", 0) + dist.get("siren_alarm", 0)

        donut_buf = self.generate_event_distribution_chart(dist)
        timeline_buf = self.generate_hourly_timeline_chart(hourly)
        incidents_buf = self.generate_security_incidents_chart(incidents)

        return {
            "distribution_chart": base64.b64encode(donut_buf.getvalue()).decode("utf-8"),
            "timeline_chart": base64.b64encode(timeline_buf.getvalue()).decode("utf-8"),
            "incidents_chart": base64.b64encode(incidents_buf.getvalue()).decode("utf-8")
        }


# Singleton instance
report_visualizer = ReportVisualizer()
