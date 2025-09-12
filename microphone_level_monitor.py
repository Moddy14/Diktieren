# -*- coding: utf-8 -*-
"""
Minimaler, wiederverwendbarer Code: Mikrofon-Audio einlesen,
Standard-Eingabegerät ermitteln und in Echtzeit einen Sprechpegel
(0–100) bereitstellen – als Signal und als kleine Qt-Widget-Komponente.

Abhängigkeiten:
  pip install sounddevice PyQt6 numpy

Einbau-Varianten:
  1) Nur Logik (MicLevelMonitor) in bestehende App integrieren.
  2) Fertiges Widget (MicLevelBar) als Drop-in verwenden.

Unter __main__ ist eine kleine Demo enthalten.
"""
from __future__ import annotations

import math
import traceback
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from PyQt6 import QtCore, QtWidgets


def _format_exc(e: Exception) -> str:
    """Formatiert Exception für Anzeige."""
    return f"{type(e).__name__}: {e}"


class MicLevelMonitor(QtCore.QObject):
    """Liest Audio vom (Standard-)Mikrofon und liefert Pegel-Updates.

    Signals
    -------
    levelChanged(int): Pegel 0–100 (glattgezogen, ca. 20 Hz)
    deviceResolved(int, str): (Geräteindex oder -1 bei unbekannt, Anzeigename)
    error(str): Fehlermeldung
    """

    levelChanged = QtCore.pyqtSignal(int)
    deviceResolved = QtCore.pyqtSignal(int, str)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
        *,
        device: Optional[int] = None,
        samplerate: Optional[int] = None,
        channels: int = 1,
        blocksize: int = 512,
        update_interval_ms: int = 50,
        scale: float = 500.0,
    ) -> None:
        super().__init__(parent)
        self._user_device = device  # Vorgabe des Aufrufers; None = System-Default
        self._samplerate = samplerate
        self._channels = max(1, int(channels))
        self._blocksize = int(blocksize)
        self._update_interval_ms = int(update_interval_ms)
        self._scale = float(scale)

        self._stream: Optional[sd.InputStream] = None
        self._resolved_device_index: int = -1
        self._resolved_device_name: str = "Unbekanntes Gerät"
        self._last_rms: float = 0.0
        self._smoothed_level: float = 0.0  # gleitender Mittelwert für ruhigere Anzeige

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer)

    # ----------------------- Public API -----------------------
    def start(self) -> None:
        """Startet den Eingabestream. Verwendet Standard-Gerät, wenn keines gesetzt."""
        try:
            self._open_stream()
            self._timer.start(self._update_interval_ms)
        except Exception as e:
            self.stop()
            self.error.emit(_format_exc(e))

    def stop(self) -> None:
        """Stoppt den Eingabestream und die Updates."""
        self._timer.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def setDevice(self, device: Optional[int]) -> None:
        """Wechselt das Gerät (None = System-Default). Läuft hot-swap-fähig."""
        self._user_device = device
        was_running = self._timer.isActive()
        self.stop()
        if was_running:
            self.start()

    def setScale(self, scale: float) -> None:
        """Skalierung für Pegel (rms*scale -> 0..100)."""
        self._scale = float(scale)

    def resolvedDevice(self) -> Tuple[int, str]:
        """Gibt das aktuell verwendete Gerät (Index oder -1) und den Namen zurück."""
        return self._resolved_device_index, self._resolved_device_name

    # --------------------- Internals --------------------------
    def _open_stream(self) -> None:
        # Gerät auflösen
        device_param = self._user_device  # None => System-Default durch PortAudio

        # Falls für das Zielgerät Informationen verfügbar sind, Samplerate/Channels konsistent setzen
        sr = self._samplerate
        ch = self._channels
        try:
            if device_param is not None:
                dev_info = sd.query_devices(device_param)
                ch = min(ch, max(1, int(dev_info.get("max_input_channels", 1))))
                if sr is None:
                    sr = int(dev_info.get("default_samplerate", 44100))
            else:
                # Keine Index-Info verfügbar – mit 44100 probieren, PortAudio nimmt Default-Gerät
                if sr is None:
                    sr = 44100
        except Exception:
            # Wenn das Abfragen des Geräts fehlschlägt, mit generischen Werten arbeiten
            if sr is None:
                sr = 44100
            ch = max(1, ch)

        # Stream anlegen
        self._stream = sd.InputStream(
            device=device_param,  # None -> Standardgerät
            channels=ch,
            samplerate=sr,
            blocksize=self._blocksize,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

        # Versuchen, Name/Index zu bestimmen (bei None nicht immer exakt möglich)
        idx = device_param if device_param is not None else self._guess_default_input_index()
        name = self._device_name_safe(idx)
        self._resolved_device_index = idx if idx is not None else -1
        self._resolved_device_name = name
        self.deviceResolved.emit(self._resolved_device_index, self._resolved_device_name)

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            # Status nur logik-intern behandeln – kein Qt aus Callback!
            pass
        if indata is None or len(indata) == 0:
            return
        # RMS des aktuellen Blocks berechnen
        rms = float(np.sqrt(np.mean(np.square(indata), dtype=np.float32)))
        self._last_rms = rms

    def _on_timer(self) -> None:
        # Pegel 0..100 aus RMS (einfaches Linear-Scaling + weiches Glätten)
        level = min(100.0, max(0.0, self._last_rms * self._scale))
        # Exponentielles Glätten, fühlbar aber reaktionsschnell
        alpha = 0.35
        self._smoothed_level = (1 - alpha) * self._smoothed_level + alpha * level
        self.levelChanged.emit(int(round(self._smoothed_level)))

    @staticmethod
    def _device_name_safe(idx: Optional[int]) -> str:
        if idx is None or idx < 0:
            return "System-Standard (PortAudio)"
        try:
            return sd.query_devices(idx).get("name", f"Gerät {idx}")
        except Exception:
            return f"Gerät {idx}"

    @staticmethod
    def _guess_default_input_index() -> Optional[int]:
        """Best-Effort-Schätzung des Default-Input-Index.
        PortAudio gibt den Default-Index nicht immer direkt preis; wir versuchen,
        sd.default.device auszulesen und plausibel zu interpretieren.
        """
        try:
            din, _ = sd.default.device
            if isinstance(din, int) and din >= 0:
                return din
        except Exception:
            pass
        # Fallback: nichts bekannt
        return None


class MicLevelBar(QtWidgets.QWidget):
    """Kleines Fertig-Widget mit Geräteschild und Progressbar."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        device: Optional[int] = None,
        samplerate: Optional[int] = None,
        channels: int = 1,
        blocksize: int = 512,
    ) -> None:
        super().__init__(parent)
        self.monitor = MicLevelMonitor(
            self,
            device=device,
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
        )

        self._lbl = QtWidgets.QLabel("🎤 Gerät: wird ermittelt…")
        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFormat("Pegel: %v%")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self._lbl)
        lay.addWidget(self._bar)
        self.setLayout(lay)

        self.monitor.deviceResolved.connect(self._on_device)
        self.monitor.levelChanged.connect(self._bar.setValue)
        self.monitor.error.connect(self._on_error)

    def start(self) -> None:
        self.monitor.start()

    def stop(self) -> None:
        self.monitor.stop()

    def setDevice(self, device: Optional[int]) -> None:
        self.monitor.setDevice(device)

    def _on_device(self, idx: int, name: str) -> None:
        suffix = f" (Index {idx})" if idx >= 0 else ""
        self._lbl.setText(f"🎤 Gerät: {name}{suffix}")

    def _on_error(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Audiofehler", msg)

    def closeEvent(self, e) -> None:
        try:
            self.stop()
        finally:
            super().closeEvent(e)


# ------------------------------ Demo ------------------------------
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = QtWidgets.QWidget()
    w.setWindowTitle("Mikrofon-Pegel (Demo)")
    layout = QtWidgets.QVBoxLayout(w)

    bar = MicLevelBar()
    layout.addWidget(bar)

    btns = QtWidgets.QHBoxLayout()
    start_btn = QtWidgets.QPushButton("Start")
    stop_btn = QtWidgets.QPushButton("Stop")
    btns.addWidget(start_btn)
    btns.addWidget(stop_btn)
    layout.addLayout(btns)

    start_btn.clicked.connect(bar.start)
    stop_btn.clicked.connect(bar.stop)

    w.resize(420, 140)
    w.show()

    # Auto-Start für Demo
    QtCore.QTimer.singleShot(0, bar.start)

    sys.exit(app.exec())


# -------------------------- Integration ---------------------------
# Beispiel-Integration in bestehendem Code:
#
#   from microphone_level_monitor import MicLevelMonitor
#
#   class MainWindow(QtWidgets.QMainWindow):
#       def __init__(self):
#           super().__init__()
#           self.monitor = MicLevelMonitor(self)
#           self.monitor.levelChanged.connect(self.on_level)
#           self.monitor.start()
#
#       def on_level(self, level: int):
#           print("Pegel:", level)
#
#   # oder direkt als Widget:
#   #   from microphone_level_monitor import MicLevelBar
#   #   widget = MicLevelBar(); widget.start()