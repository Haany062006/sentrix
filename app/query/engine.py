"""
SENTRIX Intelligent Query Engine
Processes natural language and structured queries against surveillance data stored in SQLite.
"""

import re
import time
from datetime import datetime, timedelta
from app.database.db import (
    get_connection,
    get_events_by_date,
    get_events_by_type,
    get_events_between,
    get_hourly_activity,
    get_event_type_distribution,
    get_person_activity,
    get_tamper_summary,
    get_recordings_summary,
    get_recent_events,
    get_recordings,
    get_all_persons,
    get_tamper_events
)


class QueryEngine:
    def __init__(self):
        pass

    def parse_date(self, query_text):
        """
        Extract date reference from query text (today, yesterday, YYYY-MM-DD, or None).
        """
        text = query_text.lower()
        today = datetime.now().date()

        if "today" in text:
            return today.strftime("%Y-%m-%d")
        if "yesterday" in text:
            yesterday = today - timedelta(days=1)
            return yesterday.strftime("%Y-%m-%d")

        # Check for ISO date format YYYY-MM-DD
        iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if iso_match:
            return iso_match.group(1)

        # Check for DD-MM-YYYY format
        dmy_match = re.search(r"\b(\d{2})[-/](\d{2})[-/](20\d{2})\b", text)
        if dmy_match:
            day, month, year = dmy_match.groups()
            return f"{year}-{month}-{day}"

        return None

    def query(self, user_query):
        """
        Main query handler. Analyzes query intent, retrieves data,
        and constructs a detailed answer with tabular data and metrics.
        """
        if not user_query or not user_query.strip():
            return {
                "success": False,
                "error": "Query cannot be empty."
            }

        q = user_query.strip()
        q_lower = q.lower()
        target_date = self.parse_date(q_lower)
        date_label = f"on {target_date}" if target_date else ("today" if "today" in q_lower else "in the system")

        # ----------------------------------------------------
        # 1. PERSON / FACE QUERIES
        # ----------------------------------------------------
        if any(w in q_lower for w in ["who was", "who visited", "who entered", "who came", "detected people", "who was detected"]):
            return self._handle_who_detected(target_date)

        # Specific person name search
        known_persons = get_all_persons()
        known_names = [p["name"] for p in known_persons]
        # Also check names from database events
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT person_name FROM events WHERE person_name IS NOT NULL")
        distinct_names = [r["person_name"] for r in cursor.fetchall() if r["person_name"]]
        conn.close()
        all_possible_names = list(set(known_names + distinct_names + ["dharshana"]))

        for name in all_possible_names:
            if name.lower() in q_lower:
                return self._handle_person_specific(name, target_date)

        if "unknown" in q_lower and ("person" in q_lower or "people" in q_lower or "face" in q_lower or "intruder" in q_lower):
            return self._handle_unknown_persons(target_date)

        # ----------------------------------------------------
        # 2. TAMPER & SIREN DETERRENT QUERIES (EXAMINER RECOMMENDATION 4)
        # ----------------------------------------------------
        if any(w in q_lower for w in ["tamper", "tampering", "lens covered", "camera moved", "camera blurred", "siren", "alarm"]):
            return self._handle_tamper_query(q_lower, target_date)

        # ----------------------------------------------------
        # 3. MONITORED OBJECT & MISSING TIMELINE QUERIES
        # ----------------------------------------------------
        if any(w in q_lower for w in ["object", "asset", "item", "last there", "last seen", "missing"]):
            return self._handle_object_monitor_query(q_lower, target_date)

        # ----------------------------------------------------
        # 4. RECORDINGS QUERIES
        # ----------------------------------------------------
        if any(w in q_lower for w in ["recording", "video", "footage", "clip"]):
            return self._handle_recordings_query(q_lower, target_date)

        # ----------------------------------------------------
        # 5. MOTION ACTIVITY & HOURLY TIMELINE
        # ----------------------------------------------------
        if "motion" in q_lower or "movement" in q_lower:
            return self._handle_motion_query(target_date)

        # Time range query e.g. "between 10 and 12" or "at 11"
        time_match = re.search(r"between\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*and\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", q_lower)
        if time_match:
            return self._handle_time_range_query(time_match, target_date)

        # ----------------------------------------------------
        # 6. GENERAL SUMMARY / STATISTICS
        # ----------------------------------------------------
        if any(w in q_lower for w in ["summary", "overview", "statistics", "stats", "report", "how many events", "what happened"]):
            return self._handle_general_summary(target_date)

        if any(w in q_lower for w in ["recent", "latest", "last events"]):
            return self._handle_recent_events()

        # ----------------------------------------------------
        # 7. DEFAULT / FALLBACK SEARCH
        # ----------------------------------------------------
        return self._handle_keyword_search(q, target_date)

    # --------------------------------------------------------
    # HANDLER IMPLEMENTATIONS
    # --------------------------------------------------------

    def _handle_who_detected(self, target_date):
        events = get_events_by_date(target_date) if target_date else get_events_by_type("face_recognized", limit=200)
        recognized = [e for e in events if e.get("event_type") == "face_recognized"]
        unknowns = [e for e in events if e.get("event_type") == "unknown_person"]

        person_counts = {}
        for e in recognized:
            name = e.get("person_name") or "Authorized Person"
            person_counts[name] = person_counts.get(name, 0) + 1

        if not person_counts and not unknowns:
            date_str = f"on {target_date}" if target_date else "recently"
            return {
                "success": True,
                "intent": "who_detected",
                "answer": f"No recognized people or unknown persons were logged {date_str}.",
                "metrics": {"known_detections": 0, "unknown_detections": 0, "unique_people": 0},
                "table": None,
                "suggestions": ["Show latest security events", "Tamper incidents summary", "How many recordings exist?"]
            }

        person_summary = ", ".join([f"{name} ({count} time{'s' if count>1 else ''})" for name, count in person_counts.items()])
        answer = f"Detected Personnel {('on ' + target_date) if target_date else 'recently'}: {person_summary}."
        if unknowns:
            answer += f" In addition, there were {len(unknowns)} unknown person alert(s)."

        rows = []
        for e in recognized[:25]:
            rows.append([
                e.get("timestamp", "")[-8:],
                e.get("person_name", "Unknown"),
                f"{(e.get('confidence') or 0)*100:.1f}%" if e.get("confidence") else "N/A",
                "Recognized"
            ])

        return {
            "success": True,
            "intent": "who_detected",
            "answer": answer,
            "metrics": {
                "unique_people": len(person_counts),
                "known_detections": len(recognized),
                "unknown_detections": len(unknowns)
            },
            "table": {
                "columns": ["Time", "Person", "Confidence", "Status"],
                "rows": rows
            },
            "suggestions": [
                f"When was {list(person_counts.keys())[0]} detected?" if person_counts else "List unknown persons",
                "Show security summary for today",
                "Were there any tamper events?"
            ]
        }

    def _handle_person_specific(self, name, target_date):
        activity = get_person_activity(person_name=name, date_str=target_date)
        if not activity:
            # Try fuzzy search in events table
            conn = get_connection()
            cursor = conn.cursor()
            date_clause = "AND timestamp LIKE ?" if target_date else ""
            params = [f"%{name}%", f"{target_date}%"] if target_date else [f"%{name}%"]
            cursor.execute(f"SELECT * FROM events WHERE message LIKE ? {date_clause} ORDER BY id DESC LIMIT 50", tuple(params))
            activity = [dict(r) for r in cursor.fetchall()]
            conn.close()

        count = len(activity)
        date_str = f"on {target_date}" if target_date else "in total"
        if count == 0:
            return {
                "success": True,
                "intent": "person_specific",
                "answer": f"No detections found for '{name}' {date_str}.",
                "metrics": {"detections": 0, "person": name},
                "table": None,
                "suggestions": ["Who was detected today?", "List unknown person detections", "Show latest 10 events"]
            }

        first_seen = activity[-1].get("timestamp", "").replace("T", " ")
        last_seen = activity[0].get("timestamp", "").replace("T", " ")
        confidences = [e.get("confidence") for e in activity if e.get("confidence")]
        avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else None
        conf_str = f" with an average match confidence of {avg_conf:.1f}%" if avg_conf else ""

        answer = f"'{name}' was recognized {count} time(s) {date_str}{conf_str}. Earliest detection was at {first_seen} and latest at {last_seen}."

        rows = []
        for e in activity[:20]:
            ts = e.get("timestamp", "").replace("T", " ")
            conf = e.get("confidence")
            rows.append([
                ts,
                name.capitalize(),
                f"{conf*100:.1f}%" if conf else "N/A",
                "Authorized"
            ])

        return {
            "success": True,
            "intent": "person_specific",
            "answer": answer,
            "metrics": {
                "person": name.capitalize(),
                "detections": count,
                "earliest_seen": first_seen,
                "latest_seen": last_seen,
                "avg_confidence": f"{avg_conf:.1f}%" if avg_conf else "N/A"
            },
            "table": {
                "columns": ["Timestamp", "Person", "Confidence", "Status"],
                "rows": rows
            },
            "suggestions": [
                f"Show all recordings {('for ' + target_date) if target_date else ''}",
                "Security summary for today",
                "Show tamper incidents"
            ]
        }

    def _handle_unknown_persons(self, target_date):
        events = get_events_by_type("unknown_person", limit=100, date_str=target_date)
        count = len(events)
        date_str = f"on {target_date}" if target_date else "recorded in total"

        if count == 0:
            return {
                "success": True,
                "intent": "unknown_persons",
                "answer": f"Good news: No unknown persons or intruders were detected {date_str}.",
                "metrics": {"unknown_count": 0},
                "table": None,
                "suggestions": ["Who was detected today?", "Show tamper alerts", "Security summary"]
            }

        answer = f"Alert: A total of {count} unknown person detection event(s) were logged {date_str}."
        rows = []
        for e in events[:20]:
            ts = e.get("timestamp", "").replace("T", " ")
            conf = e.get("confidence")
            rows.append([
                ts,
                "Unknown Individual",
                f"{conf*100:.1f}%" if conf else "Low",
                "ALERT"
            ])

        return {
            "success": True,
            "intent": "unknown_persons",
            "answer": answer,
            "metrics": {"unknown_count": count},
            "table": {
                "columns": ["Timestamp", "Classification", "Confidence", "Status"],
                "rows": rows
            },
            "suggestions": ["Show latest 10 events", "Who was recognized today?", "Show recordings"]
        }

    def _handle_tamper_query(self, query_text, target_date):
        tamper_summary = get_tamper_summary(target_date)
        tamper_list = get_tamper_events(limit=100)
        siren_events = get_events_by_type("siren_trigger", limit=100, date_str=target_date)
        if target_date:
            tamper_list = [t for t in tamper_list if t.get("timestamp", "").startswith(target_date)]

        total_tamper = sum(tamper_summary.values())
        siren_count = len(siren_events)
        date_str = f"on {target_date}" if target_date else "in total"

        if total_tamper == 0 and siren_count == 0:
            return {
                "success": True,
                "intent": "tamper_incidents",
                "answer": f"Camera integrity is intact. No camera tampering incidents or siren triggers were recorded {date_str}.",
                "metrics": {"total_tamper": 0, "siren_triggers": 0, "status": "CAMERA OK"},
                "table": None,
                "suggestions": ["Security summary", "Show today's recordings", "Who was detected today?"]
            }

        breakdown = ", ".join([f"{k}: {v}" for k, v in tamper_summary.items()]) if tamper_summary else "Direct Siren Triggers"
        answer = f"CRITICAL SECURITY ALERT: {total_tamper} camera tamper incident(s) and {siren_count} deterrent siren activation(s) recorded {date_str}. Breakdown: {breakdown}. Active siren alarm deters unauthorized camera sabotage (lens covered, blur, or movement)."

        rows = []
        for t in tamper_list[:25]:
            ts = t.get("timestamp", "").replace("T", " ")
            rows.append([
                ts,
                t.get("tamper_type", "Tamper"),
                f"{t.get('brightness', 0):.1f}" if t.get("brightness") is not None else "N/A",
                f"{t.get('blur_variance', 0):.1f}" if t.get("blur_variance") is not None else "N/A",
                "SIREN TRIGGERED"
            ])

        return {
            "success": True,
            "intent": "tamper_incidents",
            "answer": answer,
            "metrics": {
                "total_incidents": total_tamper,
                "siren_triggers": siren_count,
                "breakdown": tamper_summary
            },
            "table": {
                "columns": ["Timestamp", "Tamper Type", "Brightness", "Blur Variance", "Alarm Action"],
                "rows": rows
            },
            "suggestions": ["Security summary for today", "Show recordings during tamper", "Who was detected today?"]
        }

    def _handle_object_monitor_query(self, query_text, target_date):
        """
        Handles queries about monitored object presence, last seen time, and missing timeline.
        Answers questions such as:
        - "When was the object last there?"
        - "When was the object missing?"
        - "Where is the monitored object?"
        """
        from app.object_monitor import object_monitor
        status = object_monitor.get_status()

        has_object = status.get("has_object", False)
        is_present = status.get("is_present", False)
        obj_name = status.get("object_name") or "Monitored Asset"
        last_seen_ts = status.get("last_seen")
        last_seen_fmt = status.get("last_seen_formatted") or "Never recorded"
        missing_start_fmt = status.get("missing_start_formatted")
        missing_sec = status.get("missing_duration_seconds", 0.0)

        # Query database for missing object events
        events = get_events_by_type("object_alert", limit=100, date_str=target_date)
        if not events:
            all_events = get_events_by_date(target_date) if target_date else get_events_by_type("", limit=200)
            events = [
                e for e in all_events
                if "monitored object" in e.get("message", "").lower() or "missing alert" in e.get("message", "").lower()
            ]

        alert_count = len(events)
        first_missing = events[0].get("timestamp", "").replace("T", " ") if events else None
        latest_missing = events[-1].get("timestamp", "").replace("T", " ") if events else None

        # Determine elapsed time since last seen
        elapsed_str = ""
        if last_seen_ts:
            diff = max(0, int(time.time() - last_seen_ts))
            if diff < 120:
                elapsed_str = f"({diff} seconds ago)"
            elif diff < 3600:
                elapsed_str = f"({diff // 60} minutes ago)"
            else:
                elapsed_str = f"({diff // 3600} hours ago)"

        # Query structured missing intervals
        intervals = []
        try:
            from app.database.db import get_missing_intervals
            intervals = get_missing_intervals(target_date)
        except Exception:
            intervals = []

        # Formulate contextual response based on specific question
        q = query_text.lower()
        if "last there" in q or "last seen" in q or ("when was the object" in q and "missing" not in q):
            if not has_object:
                answer = "No object is currently registered for monitoring. You can upload an image of an object on the dashboard."
            elif is_present:
                answer = f"The monitored asset '{obj_name}' is currently IN VIEW (Present). It was last confirmed present in camera view at {last_seen_fmt} {elapsed_str}."
            else:
                answer = f"The monitored asset '{obj_name}' is currently MISSING from camera view! It was last confirmed present at {last_seen_fmt} {elapsed_str}."

        elif "missing" in q or "when was it missing" in q or "when was the object missing" in q:
            if not has_object and not intervals:
                answer = "No object is currently registered for monitoring. You can upload an image of an object on the dashboard."
            elif intervals:
                asset_label = intervals[0].get("asset_name") if (intervals and not has_object) else obj_name
                interval_lines = []
                for it in intervals[:5]:
                    st = it.get("start_time", "N/A")
                    et = it.get("end_time") or "Ongoing"
                    dstr = it.get("duration_str") or "In progress"
                    interval_lines.append(f"From {st} to {et} (Duration: {dstr})")
                intervals_str = "\n• " + "\n• ".join(interval_lines)
                status_str = "IN VIEW (Present)" if is_present else ("NOT DETECTED (MISSING)" if has_object else "Monitoring Inactive")
                answer = f"The monitored asset '{asset_label}' was missing during the following time period(s):{intervals_str}\nCurrent status: {status_str}."
            elif alert_count > 0:
                answer = f"The monitored asset '{obj_name}' was detected missing {alert_count} time(s). First missing alert occurred at {first_missing}, and the latest missing alert occurred at {latest_missing}. The asset was last confirmed present at {last_seen_fmt} {elapsed_str}. Current status: {'IN VIEW (Present)' if is_present else 'NOT DETECTED (MISSING)'}."
            elif not is_present:
                answer = f"The monitored asset '{obj_name}' is currently NOT DETECTED (MISSING) in camera view! It was last confirmed present at {last_seen_fmt} {elapsed_str}."
            else:
                answer = f"No missing incidents recorded for '{obj_name}'. The asset is currently IN VIEW and was last confirmed present at {last_seen_fmt} {elapsed_str}."

        else:
            # General status query
            if not has_object:
                answer = "No monitored object is currently configured. You can upload an object image from the web dashboard."
            elif intervals:
                latest_it = intervals[0]
                st = latest_it.get("start_time", "N/A")
                et = latest_it.get("end_time") or "Ongoing"
                dur = latest_it.get("duration_str") or "In progress"
                answer = f"Monitored asset '{obj_name}' status: {'IN VIEW' if is_present else 'CRITICAL: MISSING'}. Total missing incidents: {len(intervals)}. Most recent interval: From {st} to {et} ({dur}). Last confirmed in camera: {last_seen_fmt} {elapsed_str}."
            elif is_present:
                answer = f"Monitored asset '{obj_name}' is currently IN VIEW. It was last confirmed at {last_seen_fmt} {elapsed_str}. Total missing alerts logged today: {alert_count}."
            else:
                answer = f"CRITICAL: Monitored asset '{obj_name}' is currently NOT DETECTED in camera view! It was last confirmed present at {last_seen_fmt} {elapsed_str}. Total missing alerts logged today: {alert_count}."

        # Build table
        if intervals:
            table_cols = ["Monitored Asset", "Time Period (From - To)", "Duration Missing", "Status"]
            rows = []
            for it in intervals[:25]:
                st = it.get("start_time", "N/A")
                et = it.get("end_time") or "Ongoing"
                dur = it.get("duration_str") or (f"{it.get('duration_seconds', 0)}s" if it.get("duration_seconds") else "Ongoing")
                stat = it.get("status", "CURRENTLY MISSING")
                rows.append([
                    it.get("asset_name") or obj_name,
                    f"From {st} to {et}",
                    dur,
                    stat
                ])
        else:
            table_cols = ["Timestamp", "Monitored Asset", "Status", "Details"]
            rows = []
            if has_object:
                rows.append([
                    "CURRENT STATUS",
                    obj_name,
                    "IN VIEW" if is_present else "MISSING ALERT",
                    f"Last confirmed present: {last_seen_fmt} {elapsed_str}"
                ])
            for e in events[:25]:
                ts = e.get("timestamp", "").replace("T", " ")
                rows.append([
                    ts,
                    obj_name,
                    "MISSING ALERT",
                    e.get("message", "Object absent from camera view")
                ])

        return {
            "success": True,
            "intent": "monitored_object",
            "answer": answer,
            "metrics": {
                "has_object": has_object,
                "is_present": is_present,
                "object_name": obj_name,
                "last_seen": last_seen_fmt,
                "missing_alerts_count": alert_count,
                "first_missing_time": first_missing,
                "latest_missing_time": latest_missing
            },
            "table": {
                "columns": table_cols,
                "rows": rows
            },
            "suggestions": [
                "When was the object last there?",
                "When was the object missing?",
                "Security summary for today",
                "Was there any camera tampering?"
            ]
        }

    def _handle_recordings_query(self, query_text, target_date):
        summary = get_recordings_summary(target_date)
        all_rec = get_recordings()
        if target_date:
            all_rec = [r for r in all_rec if r.get("created_at", "").startswith(target_date) or target_date.replace("-", "") in r.get("filename", "")]

        count = len(all_rec)
        tot_dur = sum([r.get("duration", 0) for r in all_rec])
        max_dur = max([r.get("duration", 0) for r in all_rec]) if all_rec else 0
        date_str = f"on {target_date}" if target_date else "in total"

        if count == 0:
            return {
                "success": True,
                "intent": "recordings",
                "answer": f"No video recordings were captured {date_str}.",
                "metrics": {"count": 0, "total_duration": "0s"},
                "table": None,
                "suggestions": ["Show motion events", "Security summary", "Who was detected today?"]
            }

        mins = int(tot_dur // 60)
        secs = int(tot_dur % 60)
        dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        max_str = f"{int(max_dur)} seconds"

        answer = f"SENTRIX has saved {count} video recording(s) {date_str}, with a total surveillance footage duration of {dur_str}. The longest recording lasted {max_str}."

        rows = []
        for r in all_rec[:20]:
            ts = r.get("created_at", "").replace("T", " ")
            dur = r.get("duration", 0)
            rows.append([
                ts,
                r.get("filename", ""),
                f"{dur:.1f}s",
                "Available"
            ])

        return {
            "success": True,
            "intent": "recordings",
            "answer": answer,
            "metrics": {
                "recording_count": count,
                "total_duration": dur_str,
                "longest_recording": max_str
            },
            "table": {
                "columns": ["Created At", "Filename", "Duration", "Status"],
                "rows": rows
            },
            "suggestions": ["Who was detected today?", "Show tamper incidents", "Security summary for today"]
        }

    def _handle_motion_query(self, target_date):
        events = get_events_by_type("motion", limit=200, date_str=target_date)
        count = len(events)
        date_str = f"on {target_date}" if target_date else "recorded"

        answer = f"Detected {count} motion trigger event(s) {date_str}. Motion triggers automated video recording with 10-second post-motion buffer."
        rows = [[e.get("timestamp", "").replace("T", " "), "Motion Detected", "Recorded"] for e in events[:20]]

        return {
            "success": True,
            "intent": "motion",
            "answer": answer,
            "metrics": {"motion_events": count},
            "table": {
                "columns": ["Timestamp", "Event", "Action"],
                "rows": rows
            },
            "suggestions": ["Show recordings", "Who was recognized?", "Security summary for today"]
        }

    def _handle_time_range_query(self, match, target_date):
        date_str = target_date or datetime.now().date().strftime("%Y-%m-%d")
        h1, m1, ampm1, h2, m2, ampm2 = match.groups()
        h1 = int(h1)
        h2 = int(h2)
        if ampm1 == "pm" and h1 < 12: h1 += 12
        if ampm1 == "am" and h1 == 12: h1 = 0
        if ampm2 == "pm" and h2 < 12: h2 += 12
        if ampm2 == "am" and h2 == 12: h2 = 0

        start_iso = f"{date_str}T{h1:02d}:00:00"
        end_iso = f"{date_str}T{h2:02d}:59:59"

        events = get_events_between(start_iso, end_iso)
        count = len(events)
        answer = f"Found {count} security event(s) on {date_str} between {h1:02d}:00 and {h2:02d}:59."

        rows = []
        for e in events[:25]:
            ts = e.get("timestamp", "").replace("T", " ")[-8:]
            rows.append([
                ts,
                e.get("event_type", "").replace("_", " ").title(),
                e.get("message", "")
            ])

        return {
            "success": True,
            "intent": "time_range",
            "answer": answer,
            "metrics": {"events_in_range": count, "time_range": f"{h1:02d}:00 - {h2:02d}:59"},
            "table": {
                "columns": ["Time", "Type", "Details"],
                "rows": rows
            },
            "suggestions": ["Security summary for today", "Show tamper alerts", "Show recordings"]
        }

    def _handle_general_summary(self, target_date):
        date_str = target_date or datetime.now().date().strftime("%Y-%m-%d")
        dist = get_event_type_distribution(date_str)
        rec_sum = get_recordings_summary(date_str)
        tamper_sum = get_tamper_summary(date_str)

        total_events = sum(dist.values())
        motion_count = dist.get("motion", 0)
        face_count = dist.get("face_recognized", 0)
        unknown_count = dist.get("unknown_person", 0)
        tamper_count = dist.get("tamper", 0)
        rec_count = rec_sum.get("total_count", 0)

        answer = (
            f"SENTRIX Security Summary for {date_str}:\n"
            f"• Total Logged Events: {total_events}\n"
            f"• Motion Events: {motion_count}\n"
            f"• Authorized Faces Recognized: {face_count}\n"
            f"• Unknown Person Alerts: {unknown_count}\n"
            f"• Camera Tamper Alerts: {tamper_count}\n"
            f"• Video Recordings Captured: {rec_count}"
        )

        rows = [
            ["Motion Detections", str(motion_count), "Normal Surveillance"],
            ["Authorized Face Recognitions", str(face_count), "Secure"],
            ["Unknown Person Alerts", str(unknown_count), "Warning" if unknown_count > 0 else "Clear"],
            ["Camera Tamper Incidents", str(tamper_count), "Alert" if tamper_count > 0 else "Clear"],
            ["Video Recordings Saved", str(rec_count), f"{rec_sum.get('total_duration', 0):.0f}s total"]
        ]

        return {
            "success": True,
            "intent": "summary",
            "answer": answer,
            "metrics": {
                "date": date_str,
                "total_events": total_events,
                "motion": motion_count,
                "recognized": face_count,
                "unknown": unknown_count,
                "tamper": tamper_count,
                "recordings": rec_count
            },
            "table": {
                "columns": ["Category", "Count", "Status"],
                "rows": rows
            },
            "suggestions": [
                "Who was detected today?",
                "Show camera tamper incidents",
                "Show list of recordings"
            ]
        }

    def _handle_recent_events(self):
        recent = get_recent_events(limit=15)
        count = len(recent)
        answer = f"Retrieved latest {count} security events recorded by SENTRIX:"

        rows = []
        for e in recent:
            ts = e.get("timestamp", "").replace("T", " ")
            rows.append([
                ts,
                e.get("event_type", "").replace("_", " ").title(),
                e.get("message", "")
            ])

        return {
            "success": True,
            "intent": "recent_events",
            "answer": answer,
            "metrics": {"count": count},
            "table": {
                "columns": ["Timestamp", "Event Type", "Description"],
                "rows": rows
            },
            "suggestions": ["Security summary for today", "Who visited today?", "Show recordings"]
        }

    def _handle_keyword_search(self, query_text, target_date):
        conn = get_connection()
        cursor = conn.cursor()
        date_clause = "AND timestamp LIKE ?" if target_date else ""
        params = [f"%{query_text}%", f"{target_date}%"] if target_date else [f"%{query_text}%"]

        cursor.execute(f"SELECT * FROM events WHERE (message LIKE ? OR event_type LIKE ?) {date_clause} ORDER BY id DESC LIMIT 25",
                       (f"%{query_text}%", f"%{query_text}%") + ((f"{target_date}%",) if target_date else ()))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        count = len(rows)
        if count == 0:
            return {
                "success": True,
                "intent": "search",
                "answer": f"No records found matching query '{query_text}'.",
                "metrics": {"matches": 0},
                "table": None,
                "suggestions": ["Security summary for today", "Who was detected today?", "Show tamper incidents"]
            }

        answer = f"Found {count} event(s) matching '{query_text}':"
        table_rows = [[r.get("timestamp", "").replace("T", " "), r.get("event_type", ""), r.get("message", "")] for r in rows]

        return {
            "success": True,
            "intent": "search",
            "answer": answer,
            "metrics": {"matches": count},
            "table": {
                "columns": ["Timestamp", "Type", "Details"],
                "rows": table_rows
            },
            "suggestions": ["Security summary for today", "Who was detected today?", "Show recordings"]
        }


# Singleton instance
query_engine = QueryEngine()
