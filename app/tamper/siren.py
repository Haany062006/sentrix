"""
SENTRIX Siren Deterrent System
Triggers audible siren and visual alarm states when camera tampering (lens covered, blur, camera moved) is detected.
"""

import time
import threading
import sys

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class SirenManager:
    """
    Manages active siren state, auto-timeout, muting, and sound synthesis.
    """

    def __init__(self, default_duration=10.0, sound_enabled=True):
        self.default_duration = float(default_duration)
        self.sound_enabled = sound_enabled
        self.active = False
        self.muted = False
        self.reason = ""
        self.start_time = 0.0
        self.duration = float(default_duration)

        self._lock = threading.Lock()
        self._sound_thread = None
        self._stop_event = threading.Event()

    def trigger(self, reason="CAMERA TAMPER DETECTED", duration=None):
        """
        Activate the siren for a specified duration (in seconds).
        """
        dur = float(duration) if duration is not None else self.default_duration
        with self._lock:
            self.active = True
            self.muted = False
            self.reason = str(reason)
            self.start_time = time.time()
            self.duration = dur
            self._stop_event.clear()

        # Start audible worker if sound enabled
        if self.sound_enabled and HAS_WINSOUND:
            if self._sound_thread is None or not self._sound_thread.is_alive():
                self._sound_thread = threading.Thread(
                    target=self._siren_sound_worker,
                    daemon=True,
                    name="SentrixSirenThread"
                )
                self._sound_thread.start()

    def stop(self):
        """
        Stop the siren immediately.
        """
        with self._lock:
            self.active = False
            self.muted = False
            self.reason = ""
            self._stop_event.set()

    def mute(self):
        """
        Mute the audible siren sound while keeping alert state.
        """
        with self._lock:
            self.muted = True
            self._stop_event.set()

    def unmute(self):
        """
        Unmute the audible siren sound.
        """
        with self._lock:
            self.muted = False

    def is_active(self):
        """
        Check if siren is currently active and within duration.
        """
        with self._lock:
            if not self.active:
                return False
            elapsed = time.time() - self.start_time
            if elapsed >= self.duration:
                self.active = False
                self._stop_event.set()
                return False
            return True

    def get_status(self):
        """
        Return status dictionary of the siren system.
        """
        active = self.is_active()
        with self._lock:
            elapsed = time.time() - self.start_time if self.start_time > 0 else 0
            remaining = max(0.0, self.duration - elapsed) if active else 0.0
            return {
                "active": active,
                "muted": self.muted,
                "reason": self.reason if active else "",
                "remaining_seconds": round(remaining, 1),
                "sound_enabled": self.sound_enabled
            }

    def _siren_sound_worker(self):
        """
        Plays dual-tone oscillating emergency siren sound using winsound on Windows.
        """
        high_freq = 1350  # Hz
        low_freq = 850    # Hz
        tone_duration_ms = 120

        while not self._stop_event.is_set():
            if not self.is_active():
                break

            with self._lock:
                is_muted = self.muted

            if is_muted:
                time.sleep(0.2)
                continue

            try:
                # Rising pitch
                for freq in range(low_freq, high_freq, 100):
                    if self._stop_event.is_set() or not self.is_active():
                        break
                    winsound.Beep(freq, tone_duration_ms // 5)

                # Falling pitch
                for freq in range(high_freq, low_freq, -100):
                    if self._stop_event.is_set() or not self.is_active():
                        break
                    winsound.Beep(freq, tone_duration_ms // 5)
            except Exception:
                break


# Singleton instance
siren_manager = SirenManager()
