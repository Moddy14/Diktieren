#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import importlib
import time
import glob

# Lösche alte Logs beim Start
for old_log in glob.glob("SprachEingabe*.log"):
    try:
        os.remove(old_log)
        print(f"Deleted old log: {old_log}")
    except:
        pass

def install_package(pkg):
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-U", pkg], 
                          capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False

def ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
        return True
    except Exception:
        if install_package(pkg):
            try:
                importlib.invalidate_caches()
                importlib.import_module(name)
                return True
            except Exception:
                pass
        
        if pkg.lower() == "pyaudio":
            install_package("pipwin")
            try:
                subprocess.run([sys.executable, "-m", "pipwin", "install", "pyaudio"], 
                             capture_output=True, timeout=300)
                importlib.invalidate_caches()
                importlib.import_module("pyaudio")
                return True
            except Exception:
                return False
        return False

print("Checking dependencies...")
if not ensure("PyQt6"):
    print("PyQt6 could not be installed")
    sys.exit(1)
if not ensure("SpeechRecognition", "speech_recognition"):
    print("SpeechRecognition could not be installed")
    sys.exit(1)
if not ensure("pyaudio", "pyaudio"):
    print("PyAudio missing - Speech recognition will be limited")
if not ensure("sounddevice"):
    print("Sounddevice could not be installed")
    sys.exit(1)
if not ensure("numpy"):
    print("Numpy could not be installed")
    sys.exit(1)

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QSize, QTimer, QSettings
from PyQt6.QtGui import QTextCursor, QFont, QPalette, QColor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                            QComboBox, QSizePolicy, QStatusBar, QGroupBox,
                            QDialog, QMessageBox, QProgressBar, QListWidget, 
                            QListWidgetItem)
import speech_recognition as sr
import pyaudio
import sounddevice as sd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from microphone_level_monitor import MicLevelBar
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import QPointF

# Logging einrichten
LOG_PATH = os.path.join(os.path.dirname(__file__), "SprachEingabe.log")
logger = logging.getLogger("SprachEingabe")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
logger.info("=== App start ===")

class WaveformWidget(QWidget):
    """Widget to display audio waveform in real-time"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(150)
        self.audio_data = []
        self.max_points = 2000  # More points for longer history
        self.downsample_factor = 5  # Less aggressive downsampling for more detail
        
    def update_waveform(self, audio_chunk):
        """Update waveform with new audio data - optimized for real-time"""
        if audio_chunk is None or len(audio_chunk) == 0:
            return
            
        try:
            # Balanced downsampling for detail and performance
            chunk_flat = audio_chunk.flatten()
            
            # Simple downsampling to preserve waveform shape
            step = max(1, len(chunk_flat) // self.downsample_factor)
            downsampled = chunk_flat[::step]
            
            # Add to buffer
            self.audio_data.extend(downsampled.tolist())
            
            # Keep only last max_points for scrolling effect
            if len(self.audio_data) > self.max_points:
                self.audio_data = self.audio_data[-self.max_points:]
            
            # Immediate repaint for real-time feel
            self.update()
        except Exception as e:
            logger.debug(f"Waveform update error: {e}")
    
    def clear_waveform(self):
        """Clear the waveform display"""
        self.audio_data = []
        self.update()
    
    def paintEvent(self, event):
        """Paint the waveform - optimized for performance"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # Draw grid lines for better visualization
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        # Horizontal center line
        painter.drawLine(0, center_y, width, center_y)
        # Vertical grid lines
        for x in range(0, width, width // 10):
            painter.drawLine(x, 0, x, height)
        
        if not self.audio_data:
            return
        
        # Calculate scaling factor with some headroom
        data_array = np.array(self.audio_data)
        max_val = np.max(np.abs(data_array)) if len(data_array) > 0 else 1
        if max_val < 0.01:  # Minimum threshold to avoid noise amplification
            max_val = 0.01
        
        # Scale factor with 80% of height to leave some margin
        scale_factor = (height * 0.4) / max_val
        
        # Create path for smoother drawing
        points = []
        data_len = len(self.audio_data)
        
        # Draw waveform with improved visualization
        if data_len > 1:
            # Calculate points
            for i in range(data_len):
                x = (i / (data_len - 1)) * width
                y = center_y - (self.audio_data[i] * scale_factor)
                points.append(QPointF(x, y))
            
            # Draw waveform with subtle gradient
            gradient_color = QColor(0, 255, 100)
            
            # Draw with less aggressive fading for better visibility
            for i in range(1, len(points)):
                # Subtle fade for older samples (minimum 100 alpha)
                alpha = int(100 + 155 * (i / len(points)))
                gradient_color.setAlpha(alpha)
                painter.setPen(QPen(gradient_color, 1.5))  # Thinner line for more detail
                painter.drawLine(points[i-1], points[i])
        
        # Draw current position indicator
        if len(points) > 0:
            painter.setPen(QPen(QColor(255, 255, 0), 3))
            painter.drawEllipse(points[-1], 3, 3)

class RecognizerWorker(QObject):
    textReady = pyqtSignal(str)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    level = pyqtSignal(int)  # Audio level signal
    audioData = pyqtSignal(np.ndarray)  # Audio waveform data signal

    def __init__(self, device_index=None, language="de-DE", languages_to_try=None, saved_config=None):
        super().__init__()
        self.device_index = device_index
        self.language = language
        self.languages_to_try = languages_to_try or [
            "de-DE", "en-US", "ru-RU", "fr-FR", "es-ES"
        ]
        self._running = False
        self.recorded_data = []  # KRITISCH: Global für Callback
        self.stream = None
        self.saved_config = saved_config or {}
        self.countdown_done = False  # Track if countdown is complete
        
        # EXAKTE Parameter aus funktionierendem Test
        self.SAMPLERATE = 44100.0
        self.CHANNELS = 1
        self.BLOCKSIZE = 512
        self.DTYPE = 'float32'
        self.LATENCY = 0.034830  # Sekunden
        self.HOST_API = 'MME'  # Wichtig für Bluetooth!
        
        # Apply saved config if available
        self._apply_saved_config()
    
    def _apply_saved_config(self):
        """Apply saved device configuration if available"""
        if self.saved_config:
            try:
                if 'samplerate' in self.saved_config:
                    self.SAMPLERATE = float(self.saved_config['samplerate'])
                    logger.info(f"Using saved samplerate: {self.SAMPLERATE}")
                if 'channels' in self.saved_config:
                    self.CHANNELS = int(self.saved_config['channels'])
                    logger.info(f"Using saved channels: {self.CHANNELS}")
                if 'latency' in self.saved_config and self.saved_config['latency'] is not None:
                    self.LATENCY = float(self.saved_config['latency'])
                    logger.info(f"Using saved latency: {self.LATENCY}")
                if 'host_api' in self.saved_config:
                    self.HOST_API = self.saved_config['host_api']
                    logger.info(f"Using saved host API: {self.HOST_API}")
            except Exception as e:
                logger.warning(f"Could not apply saved config: {e}")
    
    def _device_config_key(self, device_name: str) -> str:
        """Generate config key for device"""
        return f"device_{device_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')[:50]}"

    def stop(self):
        self._running = False
        self.countdown_done = False  # Reset for next recording
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass

    def audio_callback(self, indata, frames, time, status):
        """Callback für Stream - KRITISCH: IMMER alle Daten aufnehmen!"""
        if status:
            logger.debug(f'Callback Status: {status}')
        # WICHTIG: IMMER alle Daten aufnehmen, nicht filtern!
        if indata is not None:
            self.recorded_data.append(indata.copy())
            # Send audio data to waveform in real-time - ALWAYS emit for real-time display
            # Emit immediately for real-time visualization
            try:
                self.audioData.emit(indata.copy())
            except:
                pass  # Ignore emit errors
    
    def run(self):
        self._running = True
        logger.info(f"Worker start: device_index={self.device_index}, language={self.language}")
        
        # Create recognizer für Spracherkennung
        recognizer = sr.Recognizer()
        
        # KRITISCHE ERKENNTNISSE aus Test:
        # 1. Warmup VOR Stream-Start (300ms Hardware)
        # 2. Warmup NACH Stream-Start (500ms Stabilisierung)
        # 3. Daten NUR vor Countdown leeren, NIE danach!
        # 4. ALLE Daten aufnehmen (nicht nach Lautstärke filtern)
        
        while self._running:
            try:
                # Device-Info abrufen
                device_info = sd.query_devices(self.device_index)
                device_name = device_info['name'].lower()
                
                # Spezielle Parameter für Samsung Galaxy Buds3 Pro
                if "buds3" in device_name or "buds 3" in device_name:
                    logger.info(f"Verwende EXAKTE Parameter für Samsung Galaxy Buds3 Pro: {device_info['name']}")
                    logger.info(f"Host API: {sd.query_hostapis(device_info['hostapi'])['name']}")
                    # Parameter bleiben wie in __init__ gesetzt (oder aus saved_config)
                elif not self.saved_config:  # Nur defaults verwenden wenn keine saved config
                    # Für andere Devices standard Parameter
                    self.SAMPLERATE = device_info['default_samplerate']
                    logger.info(f"Device: {device_info['name']}, Host API: {sd.query_hostapis(device_info['hostapi'])['name']}")
                else:
                    logger.info(f"Using saved config for: {device_info['name']}")
                
                # KRITISCH: Hardware-Warmup VOR Stream-Start (300ms)
                logger.info("⏳ Hardware-Warmup (300ms)...")
                self.status.emit("⏳ Mikrofon wird aktiviert...")
                time.sleep(0.3)
                
                # Stream erstellen mit Callback
                logger.info(f"Erstelle Stream: Device={self.device_index}, SR={self.SAMPLERATE}, BS={self.BLOCKSIZE}")
                self.stream = sd.InputStream(
                    device=self.device_index,
                    channels=self.CHANNELS,
                    samplerate=self.SAMPLERATE,
                    blocksize=self.BLOCKSIZE,
                    dtype=self.DTYPE,
                    latency=self.LATENCY,
                    callback=self.audio_callback
                )
                
                # Stream starten
                self.stream.start()
                logger.info("Stream gestartet")
                
                # Save successful config back to settings
                if self.device_index is not None:
                    try:
                        settings = QSettings('MikrofoneTool', 'DeviceConfigs')
                        device_name = device_info.get('name', '')
                        key = self._device_config_key(device_name)
                        settings.setValue(f"{key}_samplerate", int(self.SAMPLERATE))
                        settings.setValue(f"{key}_channels", self.CHANNELS)
                        settings.setValue(f"{key}_latency", str(self.LATENCY) if self.LATENCY else "None")
                        settings.setValue(f"{key}_host_api", sd.query_hostapis(device_info['hostapi'])['name'])
                        logger.info(f"Saved working config for {device_name}")
                    except Exception as e:
                        logger.warning(f"Could not save config: {e}")
                
                # KRITISCH: Stream-Warmup (500ms Stabilisierung)
                time.sleep(0.5)
                logger.info("Stream-Warmup abgeschlossen")
                
                # Warmup-Daten verwerfen - KRITISCH: NUR HIER!
                self.recorded_data = []
                logger.info("Warmup-Daten verworfen")
                
                # KRITISCH: 3-Sekunden Countdown
                for i in range(3, 0, -1):
                    self.status.emit(f"🔢 Countdown: {i}...")
                    logger.info(f"Countdown: {i}...")
                    time.sleep(1)
                
                # Mark countdown as done for waveform display
                self.countdown_done = True
                
                # KRITISCH: recorded_data NICHT mehr leeren nach Countdown!
                self.status.emit("🔴 AUFNAHME LÄUFT - SPRECHEN SIE JETZT!")
                logger.info("🔴 AUFNAHME GESTARTET - kontinuierliche Aufnahme aktiv")
                
                # Kontinuierliche Aufnahme-Schleife
                recording_duration = 3  # 3 Sekunden Segmente für schnellere Reaktion
                
                while self._running:
                    # Warte auf genug Daten (10 Sekunden)
                    time.sleep(recording_duration)
                    
                    # Daten sammeln - KRITISCH: recorded_data NICHT leeren!
                    if not self.recorded_data:
                        logger.debug("Keine Audio-Daten vorhanden, warte...")
                        continue
                    
                    # Audio-Daten zusammenführen
                    audio = np.concatenate(self.recorded_data) if self.recorded_data else np.array([])
                    
                    # recorded_data für nächste Aufnahme leeren
                    self.recorded_data = []
                    
                    if len(audio) == 0:
                        logger.debug("Leere Audio-Daten, überspringe...")
                        continue
                    
                    # Debug-Info
                    duration_sec = len(audio) / self.SAMPLERATE
                    max_amp = np.max(np.abs(audio))
                    avg_amp = np.mean(np.abs(audio))
                    logger.info(f"Audio-Segment: {duration_sec:.2f}s, Max: {max_amp:.4f}, Avg: {avg_amp:.6f}")
                    
                    # Send audio data to waveform widget
                    self.audioData.emit(audio)
                    
                    # Prüfe ob Audio laut genug ist
                    if max_amp < 0.001:
                        logger.debug("Audio zu leise, warte auf Sprache...")
                        self.status.emit("🔴 AUFNAHME LÄUFT - Warte auf Sprache...")
                        continue
                    
                    # Konvertiere für Spracherkennung (KRITISCH: Korrekte Konvertierung!)
                    self.status.emit("⚡ Verarbeite Audio...")
                    logger.info("Konvertiere Audio für Spracherkennung...")
                    
                    # Konvertiere float32 zu int16 für speech_recognition
                    audio_int16 = (audio * 32767).astype(np.int16)
                    
                    # Falls Stereo, zu Mono konvertieren
                    if len(audio_int16.shape) > 1 and audio_int16.shape[1] > 1:
                        audio_int16 = np.mean(audio_int16, axis=1).astype(np.int16)
                    
                    # Erstelle AudioData für speech_recognition
                    audio_data = sr.AudioData(audio_int16.tobytes(), int(self.SAMPLERATE), 2)
                    
                    # Spracherkennung
                    try:
                        if self.language == "auto":
                            recognized_text = None
                            for lang in self.languages_to_try:
                                try:
                                    recognized_text = recognizer.recognize_google(audio_data, language=lang)
                                    if recognized_text and recognized_text.strip():
                                        self.textReady.emit(recognized_text)
                                        self.status.emit("🔴 AUFNAHME LÄUFT - Sprechen Sie weiter...")
                                        logger.info(f"✅ Erkannt ({lang}): {recognized_text}")
                                        break
                                except sr.UnknownValueError:
                                    logger.debug(f"Keine Erkennung für {lang}")
                                    continue
                            if not recognized_text:
                                logger.debug("Keine Sprache erkannt")
                                self.status.emit("🔴 AUFNAHME LÄUFT - Sprechen Sie deutlicher...")
                        else:
                            text = recognizer.recognize_google(audio_data, language=self.language)
                            if text and text.strip():
                                logger.info(f"✅ Erkannt ({self.language}): {text}")
                                self.textReady.emit(text)
                                self.status.emit("🔴 AUFNAHME LÄUFT - Sprechen Sie weiter...")
                    except sr.UnknownValueError:
                        logger.debug("Konnte Sprache nicht verstehen")
                        self.status.emit("🔴 AUFNAHME LÄUFT - Sprechen Sie deutlicher...")
                    except sr.RequestError as e:
                        logger.error(f"Fehler bei Spracherkennung: {e}")
                        self.error.emit(f"Netzwerkfehler: {e}")
                        time.sleep(2)

            except Exception as e:
                # Stream-Fehler - versuche neu zu starten
                logger.error(f"Stream-Fehler: {e}")
                self.error.emit(f"Mikrofon-Fehler: {str(e)[:80]}")
                
                # Stream schließen falls offen
                if self.stream:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                    self.stream = None
                
                # Kurz warten vor Neuversuch
                if self._running:
                    time.sleep(2)
                    logger.info("Versuche Stream neu zu starten...")
                    continue
                else:
                    break
        
        # Aufräumen beim Beenden
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        logger.info("Worker beendet")
        self.finished.emit()



class DictationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Speech Recognition - Final Version")
        self.resize(1000, 700)
        self.setMinimumSize(QSize(900, 600))
        
        font = QFont("Segoe UI", 10)
        self.setFont(font)
        
        self.central = QWidget()
        self.setCentralWidget(self.central)
        
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Initialize QSettings for device configs
        self.settings = QSettings('MikrofoneTool', 'DeviceConfigs')
        
        # Device monitoring for hot-plug detection
        self.last_device_count = 0
        self.last_device_names = set()
        
        self.thread = None
        self.worker = None
        self.language = None
        self.auto_languages = [
            "de-DE", "en-US", "ru-RU", "fr-FR", "es-ES"
        ]
        
        self.setup_ui()
        
        # Initialize device tracking before first scan
        self.last_device_count = 0
        self.last_device_names = set()
        
        self.scan_devices()
        
        # Initialize with current devices to avoid duplicate scan
        try:
            devices = sd.query_devices()
            current = [d for d in devices if d.get('max_input_channels', 0) > 0]
            self.last_device_count = len(current)
            self.last_device_names = set(d.get('name', '') for d in current)
        except:
            pass
        
        # Start device monitoring (hot-plug detection)
        self.device_monitor = QTimer(self)
        self.device_monitor.timeout.connect(self.check_device_changes)
        self.device_monitor.start(3000)  # Check every 3 seconds
        
        # Aufnahme-Indicator ohne Blinken
        self.flash_timer = QTimer()
        self.flash_on = False

    def setup_ui(self):
        layout = QVBoxLayout(self.central)
        
        # Device selection
        device_group = QGroupBox("Microphone")
        device_layout = QHBoxLayout()
        
        self.mic_combo = QComboBox()
        self.mic_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mic_combo.setMinimumHeight(32)
        
        self.refresh_btn = QPushButton("Refresh")
        self.reconnect_btn = QPushButton("Reconnect")
        self.test_btn = QPushButton("Test")
        self.test_btn.setToolTip("Erweiterte Mikrofon-Tests öffnen")
        self.config_btn = QPushButton("⚙️")
        self.config_btn.setToolTip("Gespeicherte Konfigurationen verwalten")
        
        device_layout.addWidget(QLabel("Device:"))
        device_layout.addWidget(self.mic_combo, stretch=1)
        device_layout.addWidget(self.refresh_btn)
        device_layout.addWidget(self.reconnect_btn)
        device_layout.addWidget(self.test_btn)
        device_layout.addWidget(self.config_btn)
        device_group.setLayout(device_layout)
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("— Sprache wählen —")
        self.lang_combo.addItem("Deutsch + Englisch (Auto)")
        self.lang_combo.addItems([
            "German (de-DE)",
            "English (en-US)",
            "Russian (ru-RU)",
            "French (fr-FR)",
            "Spanish (es-ES)"
        ])
        # Standard auf Deutsch setzen, bevor Signale verbunden werden
        index_de = self.lang_combo.findText("German (de-DE)")
        if index_de >= 0:
            self.lang_combo.setCurrentIndex(index_de)
            self.language = "de-DE"
        
        self.start_btn = QPushButton("▶ START RECORDING")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setMinimumWidth(200)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.start_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setMinimumWidth(200)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        
        control_layout.addWidget(QLabel("Language:"))
        control_layout.addWidget(self.lang_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        # Mikrofon-Pegelanzeige
        self.mic_level_bar = MicLevelBar(self)
        self.mic_level_bar.setMaximumHeight(80)
        
        # Waveform display
        self.waveform = WaveformWidget(self)
        
        # Big status indicator
        self.status_label = QLabel("⚪ Ready")
        self.status_label.setMinimumHeight(60)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                color: #333;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 20px;
            }
        """)
        
        # Recording indicator
        self.recording_label = QLabel("")
        self.recording_label.setMinimumHeight(30)
        self.recording_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_label.hide()
        
        # Text area
        self.text = QTextEdit()
        self.text.setPlaceholderText(
            "Your text will appear here...\n\n"
            "IMPORTANT:\n"
            "1. Click START\n"
            "2. Wait for 'RECORDING' status\n"
            "3. Start speaking IMMEDIATELY\n"
            "4. Speak LOUD and CLEAR"
        )
        self.text.setStyleSheet("""
            QTextEdit {
                font-size: 16px;
                padding: 15px;
                border: 2px solid #ccc;
                border-radius: 4px;
            }
        """)
        
        # Actions
        action_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.copy_btn = QPushButton("Copy")
        
        action_layout.addWidget(self.clear_btn)
        action_layout.addWidget(self.copy_btn)
        action_layout.addStretch()
        
        # Layout
        layout.addWidget(device_group)
        layout.addLayout(control_layout)
        layout.addWidget(self.mic_level_bar)  # Pegelanzeige hinzufügen
        layout.addWidget(self.waveform)  # Waveform hinzufügen
        layout.addWidget(self.status_label)
        layout.addWidget(self.recording_label)
        layout.addWidget(self.text, stretch=1)
        layout.addLayout(action_layout)
        
        # Connect
        self.refresh_btn.clicked.connect(self.scan_devices)
        self.reconnect_btn.clicked.connect(self.attempt_reconnect)
        self.test_btn.clicked.connect(self.open_test_dialog)
        self.config_btn.clicked.connect(self.open_config_manager)
        self.start_btn.clicked.connect(self.start_dictation)
        self.stop_btn.clicked.connect(self.stop_dictation)
        self.clear_btn.clicked.connect(self.text.clear)
        self.copy_btn.clicked.connect(self.copy_text)
        self.mic_combo.currentIndexChanged.connect(self.update_start_button_state)
        # Jetzt erst Sprachsignal verbinden (verhindert Frühaufruf, wenn Buttons noch fehlen)
        self.lang_combo.currentTextChanged.connect(self.on_language_changed)

    def test_device_with_sounddevice(self, device_idx, device_info):
        """Test if a device works with sounddevice (like in Mikrophon.Test.py)"""
        try:
            # Get device details
            device = sd.query_devices(device_idx)
            samplerate = int(device.get('default_samplerate', 44100))
            max_channels = device.get('max_input_channels', 1)
            channels = min(1, max_channels)
            
            # Test configurations in order of preference
            configurations = [
                (channels, 44100, None),  # Standard for modern Bluetooth
                (channels, samplerate, None),  # Device default
                (channels, 48000, None),
                (channels, 16000, None),  # Legacy Bluetooth
                (channels, 8000, None),
            ]
            
            for ch, sr, latency in configurations:
                try:
                    # Test if configuration works (like in Mikrophon.Test.py)
                    sd.check_input_settings(device=device_idx, channels=ch, samplerate=sr)
                    
                    # Quick audio level test (0.3 seconds) to check for actual signal
                    quality = 50  # Default quality for devices that pass basic test
                    try:
                        # Try to activate the device with warmup
                        warmup_stream = sd.InputStream(
                            device=device_idx,
                            channels=ch,
                            samplerate=sr,
                            blocksize=512,
                            dtype='float32'
                        )
                        warmup_stream.start()
                        time.sleep(0.1)  # Brief warmup
                        warmup_stream.close()
                        
                        # Now record a test sample
                        recording = sd.rec(int(0.3 * sr), samplerate=sr, channels=ch,
                                         device=device_idx, dtype='float32')
                        sd.wait()
                        
                        # Check if device has any signal
                        max_val = np.max(np.abs(recording))
                        
                        # Don't reject devices with low signal - Bluetooth devices often start quiet
                        if max_val < 0.00001:  # Extremely low signal
                            # Device might be Bluetooth that needs activation
                            quality = 25  # Low quality score but still usable
                            logger.debug(f"Device {device_idx} has very low signal (max={max_val:.6f}), may need warmup")
                        elif max_val < 0.0001:
                            quality = 35  # Moderate-low quality
                            logger.debug(f"Device {device_idx} has low signal (max={max_val:.6f})")
                        elif max_val > 0:
                            # Calculate quality percentage (improved scale)
                            # Scale: 0.0001 = 35%, 0.001 = 50%, 0.01 = 70%, 0.1 = 90%, 1.0 = 100%
                            if max_val >= 0.1:
                                quality = 90
                            elif max_val >= 0.01:
                                quality = 70
                            elif max_val >= 0.001:
                                quality = 50
                            else:
                                quality = 40
                        
                        # Additional test: Try to open a stream (catches WDM-KS failures)
                        # This is crucial for devices that appear to work but fail when actually used
                        try:
                            test_stream = sd.InputStream(
                                device=device_idx,
                                channels=ch,
                                samplerate=sr,
                                blocksize=512,
                                dtype='float32'
                            )
                            test_stream.close()
                            logger.debug(f"Device {device_idx} works with {sr}Hz, {ch}ch (signal={max_val:.4f}, quality={quality}%, stream OK)")
                        except Exception as stream_err:
                            err_str = str(stream_err)
                            # Check for various WDM-KS related errors that indicate the device won't work
                            if ('Invalid device' in err_str or 
                                'PaErrorCode -9996' in err_str or
                                'Windows WDM-KS error' in err_str or
                                'PaErrorCode -9999' in err_str or
                                'Blocking API not supported' in err_str):
                                logger.debug(f"Device {device_idx} has WDM-KS issues, skipping: {stream_err}")
                                return False, None, None, 0
                            # Other stream errors might be recoverable
                            logger.debug(f"Device {device_idx} stream test warning: {stream_err}")
                        
                        return True, sr, ch, quality
                        
                    except Exception as rec_err:
                        # Recording failed - check error type
                        err_str = str(rec_err)
                        if 'WDM-KS error' in err_str or 'DeviceIoControl' in err_str:
                            # Device not properly connected/available
                            logger.debug(f"Device {device_idx} not available (WDM-KS error), skipping")
                            return False, None, None, 0
                        
                        # Other errors - might still work with warmup
                        logger.debug(f"Recording test skipped for device {device_idx}: {rec_err}")
                        logger.debug(f"Device {device_idx} tentatively accepted at {sr}Hz, {ch}ch")
                        return True, sr, ch, 45  # Default quality for untested devices (slightly below average)
                        
                except Exception as e:
                    logger.debug(f"Config {sr}Hz failed for device {device_idx}: {e}")
                    continue
            
            return False, None, None, 0
        except Exception as e:
            logger.debug(f"Sounddevice test failed for device {device_idx}: {e}")
            return False, None, None, 0
    
    def scan_devices(self):
        logger.info("Scanning devices with hybrid approach")
        self.status_label.setText("🔍 Scanning devices...")
        QApplication.processEvents()

        # Clear sounddevice cache
        try:
            sd._terminate()
            sd._initialize()
        except:
            pass

        # Get default input device
        default_input_index = None
        try:
            default_input_index, _ = sd.default.device
            if isinstance(default_input_index, str):
                default_input_index = None  # Ignore if it's a string
            logger.info(f"Default input device index: {default_input_index}")
        except:
            pass

        try:
            # Get all devices from both libraries
            sr_devices = sr.Microphone.list_microphone_names()
            sd_devices = sd.query_devices()
        except Exception as e:
            logger.exception("Error listing devices")
            self.status_label.setText(f"❌ Error: {e}")
            self.mic_combo.clear()
            self.update_start_button_state()
            return

        self.mic_combo.clear()
        self.device_configs = {}  # Store working configs
        candidate_indices = []
        preferred_index = -1
        tested_devices = set()
        failed_devices = []  # Track devices that fail tests
        working_devices = []  # Store devices with quality for sorting
        default_device_combo_index = -1  # Track where default device ends up in combo
        
        # Test devices using sounddevice first (more reliable)
        for idx, device in enumerate(sd_devices):
            if device.get('max_input_channels', 0) > 0:
                name = device.get('name', '')
                lower = name.lower()
                
                # Filter out likely non-functional devices
                # Skip generic "Line ()" and "Input" devices without proper names
                if name == "Line ()" or (name.startswith("Input (") and "hands-free" not in lower):
                    logger.debug(f"Skipping likely non-functional device {idx}: {name}")
                    continue
                    
                # Test remaining devices
                if True:  # Teste alle anderen Geräte
                    logger.debug(f"Testing device {idx}: {name}")
                    
                    # Test with sounddevice
                    works, sample_rate, channels, quality = self.test_device_with_sounddevice(idx, device)
                    
                    if works:
                        # Store working configuration
                        self.device_configs[idx] = {
                            'sample_rate': sample_rate,
                            'channels': channels,
                            'name': name,
                            'quality': quality
                        }
                        
                        # Collect devices for sorting by quality
                        working_devices.append({
                            'idx': idx,
                            'name': name,
                            'lower': lower,
                            'sample_rate': sample_rate,
                            'quality': quality,
                            'is_default': idx == default_input_index
                        })
                        
                        tested_devices.add(name)
                        logger.info(f"Device {idx} added: {name} @ {sample_rate}Hz, quality={quality}%")
                    else:
                        failed_devices.append(f"{idx}: {name}")
                        logger.warning(f"Device {idx} ({name}) failed sounddevice test")
        
        # Sort devices by quality (highest first)
        working_devices.sort(key=lambda x: x['quality'], reverse=True)
        
        # Add sorted devices to combo box
        for dev in working_devices:
            idx = dev['idx']
            name = dev['name']
            sample_rate = dev['sample_rate']
            quality = dev['quality']
            lower = dev['lower']
            
            # Samsung Galaxy Buds3 Pro priority
            if "buds3 pro" in lower or ("kopfhörer" in lower and "buds3" in lower):
                preferred_index = len(candidate_indices)
            elif "buds" in lower and "hands-free" in lower:
                if preferred_index == -1:
                    preferred_index = len(candidate_indices)
            
            # Include quality in display
            quality_str = f" {quality}%" if quality > 0 else ""
            display = name if len(name) < 50 else name[:47] + "..."
            self.mic_combo.addItem(f"{idx}: {display} [{sample_rate}Hz]{quality_str}", idx)
            
            # Check if this is the default device
            if dev['is_default']:
                default_device_combo_index = self.mic_combo.count() - 1
                logger.info(f"Found default device at combo index {default_device_combo_index}")
            
            candidate_indices.append(idx)
        
        # Fallback: Test remaining SR devices not found by sounddevice
        for i, name in enumerate(sr_devices):
            if name not in tested_devices:
                lower = (name or "").lower()
                # Skip obvious output devices
                if any(x in lower for x in ['output', 'speaker', 'lautsprecher', 'spdif', 'nvidia', 'realtek digital']):
                    continue
                    
                # Test potential input devices
                try:
                    # Quick SR test with proper error handling
                    test_mic = None
                    try:
                        test_mic = sr.Microphone(device_index=i)
                    except Exception:
                        # Microphone initialization failed
                        continue
                    
                    if test_mic:
                        try:
                            with test_mic as source:
                                # Just check if we can open it
                                if hasattr(source, 'stream') and source.stream is not None:
                                    display = f"[SR] {name}" if len(name) < 50 else f"[SR] {name[:47]}..."
                                    self.mic_combo.addItem(f"{i}: {display}", i)
                                    candidate_indices.append(i)
                                    logger.debug(f"SR device {i} added as fallback")
                        except Exception:
                            # Context manager failed
                            pass
                except Exception as e:
                    logger.debug(f"SR device {i} ({name}) failed: {e}")

        if self.mic_combo.count() > 0:
            # Priority: 1. Default device, 2. Preferred device (Buds3), 3. First device
            if default_device_combo_index != -1:
                self.mic_combo.setCurrentIndex(default_device_combo_index)
                logger.info(f"Selected default device at combo index {default_device_combo_index}")
            elif preferred_index != -1:
                self.mic_combo.setCurrentIndex(preferred_index)
                logger.info(f"Selected preferred device at combo index {preferred_index}")
            else:
                self.mic_combo.setCurrentIndex(0)
                logger.info("Selected first available device")
            logger.info(f"Found {self.mic_combo.count()} relevant device(s)")
            self.status_label.setText(f"✅ Found {self.mic_combo.count()} relevant device(s)")
        else:
            logger.warning(f"No working devices found. {len(failed_devices)} devices failed tests.")
            failed_list = "\n".join(failed_devices[:5])  # Show first 5
            self.status_label.setText(f"❌ No working devices. Failed: {len(failed_devices)}")
            if failed_devices:
                logger.error(f"Failed devices:\n{failed_list}")

        # Log summary
        if failed_devices:
            logger.info(f"Scan complete: {self.mic_combo.count()} working, {len(failed_devices)} failed")
        
        self.update_start_button_state()



    def on_language_changed(self, text):
        if "Deutsch + Englisch" in text:
            self.language = "auto"
            self.auto_languages = ["de-DE", "en-US"]
        elif "Auto" in text:
            self.language = "auto"
        elif "— Sprache wählen —" in text or not text:
            self.language = None
        elif "de-DE" in text:
            self.language = "de-DE"
        elif "en-US" in text:
            self.language = "en-US"
        elif "ru-RU" in text:
            self.language = "ru-RU"
        elif "fr-FR" in text:
            self.language = "fr-FR"
        elif "es-ES" in text:
            self.language = "es-ES"
        else:
            self.language = None
        self.update_start_button_state()

    def _device_config_key(self, device_name: str) -> str:
        """Generate unique key for device configuration"""
        return f"device_{device_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')[:50]}"
    
    def load_saved_config_for_current_device(self):
        """Load saved configuration for the currently selected device"""
        idx = self.mic_combo.currentData()
        if idx is None:
            return None
        try:
            info = sd.query_devices(idx)
            name = info.get('name', 'Unknown')
            key = self._device_config_key(name)
            
            # Check if we have saved config for this device
            if self.settings.contains(f"{key}_samplerate"):
                sr = self.settings.value(f"{key}_samplerate", type=int)
                ch = self.settings.value(f"{key}_channels", type=int)
                lat_str = self.settings.value(f"{key}_latency", type=str)
                host_api = self.settings.value(f"{key}_host_api", type=str)
                
                latency = None
                if lat_str and lat_str not in ('None', 'null', ''):
                    try:
                        latency = float(lat_str)
                    except:
                        latency = None
                
                config = {
                    "samplerate": sr,
                    "channels": ch,
                    "latency": latency,
                    "host_api": host_api
                }
                logger.info(f"Loaded saved config for {name}: {config}")
                return config
        except Exception as e:
            logger.warning(f"Could not load saved config: {e}")
        return None
    
    def check_device_changes(self):
        """Check for device hot-plug/unplug events"""
        try:
            devices = sd.query_devices()
            current = [d for d in devices if d.get('max_input_channels', 0) > 0]
            count = len(current)
            names = set(d.get('name', '') for d in current)
            
            if count != self.last_device_count or names != self.last_device_names:
                added = names - self.last_device_names
                removed = self.last_device_names - names
                
                self.last_device_count = count
                self.last_device_names = names
                
                # Only refresh if there were actual changes and not during recording
                if (added or removed) and self.worker is None:
                    # Show notification
                    messages = []
                    if added:
                        messages.append(f"🔌 Neu: {list(added)[0][:30]}")
                    if removed:
                        messages.append(f"🔍 Entfernt: {list(removed)[0][:30]}")
                    
                    if messages:
                        self.status_label.setText(" | ".join(messages))
                        logger.info(f"Device changes detected - Added: {added}, Removed: {removed}")
                        
                        # Refresh device list
                        self.scan_devices()
                        
                        # Clear notification after 5 seconds
                        QTimer.singleShot(5000, lambda: self.status_label.setText("⚪ Ready"))
        except Exception as e:
            logger.debug(f"Error checking device changes: {e}")
    
    def open_test_dialog(self):
        """Open test dialog with options for single or batch test"""
        # Create selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Mikrofon-Tests")
        dialog.setMinimumSize(400, 200)
        
        layout = QVBoxLayout()
        
        # Info
        info_label = QLabel("Wählen Sie den gewünschten Test:")
        info_label.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(info_label)
        
        # Single test button
        single_btn = QPushButton("🎤 Einzeltest - Aktuelles Gerät")
        single_btn.setMinimumHeight(40)
        single_btn.clicked.connect(lambda: self.run_single_test(dialog))
        layout.addWidget(single_btn)
        
        # Batch test button
        batch_btn = QPushButton("📊 Batch-Test - Alle Geräte")
        batch_btn.setMinimumHeight(40)
        batch_btn.clicked.connect(lambda: self.run_batch_test(dialog))
        layout.addWidget(batch_btn)
        
        # Cancel button
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(dialog.close)
        layout.addWidget(cancel_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def run_single_test(self, parent_dialog):
        """Run test for single selected device"""
        parent_dialog.close()
        
        device_index = self.mic_combo.currentData()
        if device_index is None:
            QMessageBox.warning(self, "Kein Gerät", "Bitte wählen Sie zuerst ein Mikrofon aus.")
            return
        
        try:
            device_info = sd.query_devices(device_index)
            device_name = device_info.get('name', 'Unknown')
            
            # Single device test dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Mikrofon-Test: {device_name}")
            dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout()
            
            # Info
            info_label = QLabel(f"🎤 Gerät: {device_name}\nIndex: {device_index}")
            layout.addWidget(info_label)
            
            # Test area
            test_text = QTextEdit()
            test_text.setReadOnly(True)
            layout.addWidget(test_text)
            
            # Test button
            def run_test():
                test_text.clear()
                test_text.append("🔄 Test läuft...\n")
                test_text.append("Sprechen Sie jetzt für 3 Sekunden!\n")
                QApplication.processEvents()
                
                try:
                    # Test recording with speech recognition
                    duration = 3
                    samplerate = int(device_info.get('default_samplerate', 44100))
                    test_text.append(f"📼 Aufnahme für {duration} Sekunden bei {samplerate}Hz...")
                    QApplication.processEvents()
                    
                    recording = sd.rec(int(duration * samplerate),
                                     samplerate=samplerate,
                                     channels=1,
                                     device=device_index,
                                     dtype='float32')
                    sd.wait()
                    
                    max_val = np.max(np.abs(recording))
                    avg_val = np.mean(np.abs(recording))
                    
                    test_text.append(f"\n✅ Test abgeschlossen!")
                    test_text.append(f"Max. Amplitude: {max_val:.4f}")
                    test_text.append(f"Durchschn. Amplitude: {avg_val:.6f}")
                    
                    if max_val > 0.001:
                        test_text.append("\n🎉 Mikrofon funktioniert!")
                        
                        # Try speech recognition
                        test_text.append("\n🔊 Versuche Spracherkennung...")
                        QApplication.processEvents()
                        
                        try:
                            import speech_recognition as sr
                            # Convert to int16 for speech recognition
                            audio_int16 = (recording * 32767).astype(np.int16)
                            
                            # Create AudioData object
                            audio = sr.AudioData(
                                audio_int16.tobytes(),
                                int(samplerate),
                                2  # 2 bytes per sample
                            )
                            
                            # Try recognition
                            r = sr.Recognizer()
                            text = r.recognize_google(audio, language="de-DE")
                            test_text.append(f"📝 Erkannt: \"{text}\"")
                        except Exception as e:
                            test_text.append("⚠️ Keine Sprache erkannt (zu leise oder undeutlich)")
                        
                        # Save successful config
                        key = self._device_config_key(device_name)
                        self.settings.setValue(f"{key}_samplerate", samplerate)
                        self.settings.setValue(f"{key}_channels", 1)
                        self.settings.setValue(f"{key}_latency", "None")
                        host_api = sd.query_hostapis()[device_info['hostapi']]['name']
                        self.settings.setValue(f"{key}_host_api", host_api)
                        test_text.append(f"\n💾 Konfiguration gespeichert")
                    else:
                        test_text.append("\n⚠️ Kein oder sehr leises Signal erkannt")
                        
                except Exception as e:
                    test_text.append(f"\n❌ Fehler: {str(e)}")
            
            test_btn = QPushButton("🎤 Test starten")
            test_btn.clicked.connect(run_test)
            layout.addWidget(test_btn)
            
            # Close button
            close_btn = QPushButton("Schließen")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Konnte Test-Dialog nicht öffnen: {str(e)}")
    
    def run_batch_test(self, parent_dialog):
        """Run batch test for all devices"""
        parent_dialog.close()
        
        # Get all input devices
        devices = []
        for i in range(self.mic_combo.count()):
            idx = self.mic_combo.itemData(i)
            if idx is not None:
                try:
                    info = sd.query_devices(idx)
                    if info.get('max_input_channels', 0) > 0:
                        devices.append({'index': idx, 'name': self.mic_combo.itemText(i)})
                except:
                    pass
        
        if not devices:
            QMessageBox.warning(self, "Keine Geräte", "Keine Eingabegeräte gefunden.")
            return
        
        # Batch test dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("📊 Batch-Test aller Mikrofone")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        # Info
        info_label = QLabel(f"Teste {len(devices)} Geräte (je 1.5 Sekunden)")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Progress
        progress = QProgressBar()
        progress.setMaximum(len(devices))
        layout.addWidget(progress)
        
        # Current device
        current_label = QLabel("Bereit...")
        layout.addWidget(current_label)
        
        # Results
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        layout.addWidget(results_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        test_results = []
        current_index = [0]  # Use list to allow modification in nested function
        
        def test_next():
            if current_index[0] >= len(devices):
                # Testing complete
                current_label.setText("✅ Test abgeschlossen!")
                results_text.append("\n" + "="*50)
                results_text.append("ZUSAMMENFASSUNG:")
                results_text.append("="*50)
                
                # Sort by quality
                test_results.sort(key=lambda x: x['quality'], reverse=True)
                
                # Count working/non-working
                working = sum(1 for r in test_results if r['quality'] > 30)
                low_signal = sum(1 for r in test_results if 0 < r['quality'] <= 30)
                no_signal = sum(1 for r in test_results if r['quality'] == 0)
                
                # Show summary
                results_text.append(f"\n📊 Statistik: {working} gut | {low_signal} schwach | {no_signal} kein Signal\n")
                
                for result in test_results:
                    status = "✅" if result['quality'] > 30 else "⚠️" if result['quality'] > 0 else "❌"
                    results_text.append(f"{status} {result['name']}: Qualität {result['quality']}%")
                
                # Save best config
                if test_results and test_results[0]['quality'] > 30:
                    best = test_results[0]
                    results_text.append(f"\n🏆 Bestes Gerät: {best['name']}")
                    
                    # Ask user if they want to use the best device
                    reply = QMessageBox.question(dialog, "Bestes Gerät verwenden?",
                        f"Möchten Sie '{best['name']}' als aktives Gerät auswählen?\n\n"
                        f"Qualität: {best['quality']}%",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes)
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        # Select best device in combo
                        for i in range(self.mic_combo.count()):
                            if self.mic_combo.itemData(i) == best['index']:
                                self.mic_combo.setCurrentIndex(i)
                                results_text.append("✅ Gerät ausgewählt!")
                                break
                
                # Remove non-working devices from list
                if no_signal > 0:
                    results_text.append(f"\n🗑️ Entferne {no_signal} Geräte ohne Signal aus der Liste...")
                    
                    # Collect indices to remove
                    removed_count = 0
                    for result in test_results:
                        if result['quality'] == 0:
                            # Find and remove from combo box
                            for i in range(self.mic_combo.count() - 1, -1, -1):
                                if self.mic_combo.itemData(i) == result['index']:
                                    self.mic_combo.removeItem(i)
                                    removed_count += 1
                                    break
                    
                    results_text.append(f"✅ {removed_count} nicht funktionierende Geräte entfernt")
                    logger.info(f"Removed {removed_count} non-working devices from list")
                
                return
            
            device = devices[current_index[0]]
            current_label.setText(f"🎤 Teste: {device['name']}...")
            progress.setValue(current_index[0])
            
            try:
                # Quick test with warmup
                info = sd.query_devices(device['index'])
                sr = int(info.get('default_samplerate', 44100))
                
                # Short warmup for Bluetooth devices
                if 'bluetooth' in device['name'].lower() or 'buds' in device['name'].lower():
                    QApplication.processEvents()
                    time.sleep(0.2)  # 200ms warmup for BT
                
                # Record for 1.5 seconds (shorter is enough for level test)
                recording = sd.rec(int(1.5 * sr), samplerate=sr, channels=1,
                                 device=device['index'], dtype='float32')
                sd.wait()
                
                max_val = np.max(np.abs(recording))
                avg_val = np.mean(np.abs(recording))
                
                # Better quality calculation
                # Max amplitude: 0.001 = 0%, 0.1 = 100%
                if max_val < 0.001:
                    quality = 0
                elif max_val > 0.1:
                    quality = 100
                else:
                    # Logarithmic scale for better distribution
                    quality = int((np.log10(max_val * 1000) + 3) * 33.33)
                    quality = max(0, min(100, quality))
                
                result = {
                    'name': device['name'],
                    'index': device['index'],
                    'max': max_val,
                    'avg': avg_val,
                    'quality': quality
                }
                test_results.append(result)
                
                status = "✅ OK" if quality > 30 else "⚠️ Leise" if quality > 0 else "❌ Kein Signal"
                results_text.append(f"{device['name']}: {status} (Qualität: {quality}%)")
                
            except Exception as e:
                test_results.append({
                    'name': device['name'],
                    'index': device['index'],
                    'quality': 0
                })
                results_text.append(f"{device['name']}: ❌ Fehler - {str(e)[:50]}")
            
            current_index[0] += 1
            QTimer.singleShot(100, test_next)
        
        start_btn = QPushButton("🚀 Test starten")
        start_btn.clicked.connect(lambda: (start_btn.setEnabled(False), test_next()))
        btn_layout.addWidget(start_btn)
        
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec()
    
    def update_start_button_state(self):
        # Abwehr gegen Aufrufe während UI-Aufbau
        if not hasattr(self, 'start_btn'):
            return
        has_device = self.mic_combo.count() > 0 and self.mic_combo.currentData() is not None
        has_language = self.language is not None
        self.start_btn.setEnabled(has_device and has_language and self.worker is None)

    def start_dictation(self):
        logger.info("Start dictation requested")
        if self.worker is not None:
            return
        
        device_index = self.mic_combo.currentData()
        if device_index is None:
            logger.warning("Start denied - no device selected")
            return
        
        # Start mic level monitor with selected device
        self.mic_level_bar.setDevice(device_index)
        self.mic_level_bar.start()
        
        # Load saved config for the selected device
        saved_config = self.load_saved_config_for_current_device()
        
        if self.language == "auto":
            self.worker = RecognizerWorker(device_index=device_index, language="auto", languages_to_try=self.auto_languages, saved_config=saved_config)
        else:
            self.worker = RecognizerWorker(device_index=device_index, language=self.language, saved_config=saved_config)
        
        # Pass device configs to worker if available
        if hasattr(self, 'device_configs'):
            self.worker.parent = lambda: self
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.textReady.connect(self.on_text_ready)
        self.worker.status.connect(self.on_status)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.level.connect(self.on_level)
        self.worker.audioData.connect(self.waveform.update_waveform)
        
        logger.info("Thread started")
        self.thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.mic_combo.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        
        # Statische Aufnahmeanzeige
        self.recording_label.setText("🔴 RECORDING IN PROGRESS - SPEAK NOW!")
        self.recording_label.setStyleSheet("""
                QLabel {
                    background-color: #ff0000;
                    color: white;
                    font-weight: bold;
                    padding: 5px;
                    border-radius: 4px;
                }
            """)
        self.recording_label.show()

    def stop_dictation(self):
        logger.info("Stop dictation requested")
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("⏸ Stopping...")
            self.flash_timer.stop()
            self.recording_label.hide()
        
        # Stop mic level monitor
        self.mic_level_bar.stop()
        
        # Clear waveform
        self.waveform.clear_waveform()

    def flash_recording(self):
        """Disabled blinking - keep static label."""
        pass

    def on_text_ready(self, text):
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Add space if needed
        current = self.text.toPlainText()
        if current and not current.endswith((" ", "\n")):
            cursor.insertText(" ")
        
        cursor.insertText(text.strip())
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()
        
        # Flash green briefly
        self.text.setStyleSheet("""
            QTextEdit {
                font-size: 16px;
                padding: 15px;
                border: 2px solid #4CAF50;
                border-radius: 4px;
                background-color: #f1f8e9;
            }
        """)
        QTimer.singleShot(500, lambda: self.text.setStyleSheet("""
            QTextEdit {
                font-size: 16px;
                padding: 15px;
                border: 2px solid #ccc;
                border-radius: 4px;
            }
        """))

    def on_status(self, status):
        self.status_label.setText(status)
        
        if "RECORDING" in status:
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #ff5252;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 20px;
                }
            """)
        elif "Processing" in status:
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #2196F3;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 20px;
                }
            """)
        elif "Calibrat" in status:
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #FFC107;
                    color: black;
                    padding: 15px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 20px;
                }
            """)

    def on_level(self, level):
        """Visual feedback for audio level"""
        pass  # Could add level meter here

    def on_error(self, error):
        self.status_label.setText(f"❌ {error}")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 20px;
            }
        """)
        logger.warning(f"Error displayed to user: {error}")
        
        # Implement fallback strategies
        error_lower = error.lower()
        
        # Check for common errors and offer solutions
        if "invalid device" in error_lower or "device error" in error_lower:
            self.handle_device_error()
        elif "timeout" in error_lower or "timed out" in error_lower:
            self.handle_timeout_error()
        elif "no internet" in error_lower or "connection" in error_lower:
            self.handle_connection_error()
    
    def handle_device_error(self):
        """Handle device errors with automatic fallback"""
        logger.info("Handling device error - attempting fallback")
        
        # Stop current recording
        self.stop_dictation()
        
        # Try to find alternative device
        current_idx = self.mic_combo.currentIndex()
        if self.mic_combo.count() > 1:
            # Switch to next device
            next_idx = (current_idx + 1) % self.mic_combo.count()
            self.mic_combo.setCurrentIndex(next_idx)
            
            # Show notification
            device_name = self.mic_combo.currentText()
            QMessageBox.information(self, "Device Switch", 
                f"Switched to alternative device:\n{device_name}\n\nClick Start to continue.")
            logger.info(f"Switched to device {device_name}")
        else:
            # No alternative devices
            QMessageBox.warning(self, "No Alternative Devices",
                "No alternative devices available.\nPlease check your microphone connections.")
    
    def handle_timeout_error(self):
        """Handle timeout errors"""
        logger.info("Handling timeout error")
        
        # Suggest reducing timeout or checking microphone
        reply = QMessageBox.question(self, "Timeout Error",
            "Speech recognition timed out.\n\n"
            "This could mean:\n"
            "• The microphone is too quiet\n"
            "• Background noise is too high\n"
            "• The threshold needs adjustment\n\n"
            "Would you like to open the Test dialog?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.open_test_dialog()
    
    def handle_connection_error(self):
        """Handle connection errors"""
        logger.info("Handling connection error")
        
        QMessageBox.warning(self, "Connection Error",
            "Could not connect to speech recognition service.\n\n"
            "Please check:\n"
            "• Internet connection\n"
            "• Firewall settings\n"
            "• Proxy configuration\n\n"
            "The app will work again once connection is restored.")
    
    def open_config_manager(self):
        """Open configuration management dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🔧 Konfigurationen verwalten")
        dialog.setModal(True)
        dialog.resize(600, 400)
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel("Hier können Sie gespeicherte Gerätekonfigurationen verwalten:")
        layout.addWidget(info_label)
        
        # Config list
        config_list = QListWidget()
        config_list.setAlternatingRowColors(True)
        
        # Load saved configs
        all_keys = self.settings.allKeys()
        devices = {}
        
        for key in all_keys:
            if '_samplerate' in key:
                device_key = key.replace('_samplerate', '')
                sr = self.settings.value(f"{device_key}_samplerate", type=int)
                ch = self.settings.value(f"{device_key}_channels", type=int)
                host = self.settings.value(f"{device_key}_host_api", "Unknown")
                
                # Clean up device name for display
                display_name = device_key.replace('device_', '').replace('_', ' ')
                if len(display_name) > 50:
                    display_name = display_name[:47] + "..."
                
                devices[device_key] = {
                    'display': display_name,
                    'sr': sr,
                    'ch': ch,
                    'host': host
                }
        
        if devices:
            for key, info in devices.items():
                item_text = f"📱 {info['display']}\n   → {info['sr']}Hz, {info['ch']}ch, {info['host']}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, key)
                config_list.addItem(item)
        else:
            config_list.addItem("Keine gespeicherten Konfigurationen vorhanden")
        
        layout.addWidget(config_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        # Delete selected button
        delete_btn = QPushButton("🗑️ Ausgewählte löschen")
        def delete_selected():
            item = config_list.currentItem()
            if item:
                key = item.data(Qt.ItemDataRole.UserRole)
                if key:
                    reply = QMessageBox.question(dialog, "Konfiguration löschen",
                        f"Möchten Sie die Konfiguration für\n'{devices[key]['display']}'\nwirklich löschen?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        # Remove all related settings
                        self.settings.remove(f"{key}_samplerate")
                        self.settings.remove(f"{key}_channels")
                        self.settings.remove(f"{key}_latency")
                        self.settings.remove(f"{key}_host_api")
                        self.settings.sync()
                        
                        # Remove from list
                        config_list.takeItem(config_list.row(item))
                        del devices[key]
                        
                        if not devices:
                            config_list.addItem("Keine gespeicherten Konfigurationen vorhanden")
                        
                        logger.info(f"Deleted config for {key}")
        
        delete_btn.clicked.connect(delete_selected)
        btn_layout.addWidget(delete_btn)
        
        # Clear all button
        clear_all_btn = QPushButton("🗑️ Alle löschen")
        def clear_all():
            if devices:
                reply = QMessageBox.question(dialog, "Alle Konfigurationen löschen",
                    f"Möchten Sie wirklich ALLE {len(devices)} gespeicherten Konfigurationen löschen?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.settings.clear()
                    self.settings.sync()
                    config_list.clear()
                    config_list.addItem("Keine gespeicherten Konfigurationen vorhanden")
                    devices.clear()
                    logger.info("Cleared all device configs")
        
        clear_all_btn.clicked.connect(clear_all)
        btn_layout.addWidget(clear_all_btn)
        
        # Close button
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(dialog.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Stats label
        stats_label = QLabel(f"📊 {len(devices)} Konfiguration(en) gespeichert")
        layout.addWidget(stats_label)
        
        dialog.setLayout(layout)
        dialog.exec()

    def on_finished(self):
        if self.thread:
            self.thread.quit()
            if not self.thread.wait(3000):
                self.thread.terminate()
            self.thread = None
        
        self.worker = None
        
        # Stop mic level monitor
        self.mic_level_bar.stop()
        
        # Clear waveform
        self.waveform.clear_waveform()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.mic_combo.setEnabled(True)
        self.lang_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.update_start_button_state()
        
        self.status_label.setText("⚪ Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e0e0e0;
                color: #333;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 20px;
            }
        """)
        
        self.recording_label.hide()
        logger.info("Stopped")

    def attempt_reconnect(self):
        logger.info("Manual reconnect requested")
        restarting = self.worker is not None
        if restarting:
            self.stop_dictation()
            # kurzen Moment warten und neu starten
            QTimer.singleShot(500, self.start_dictation)
        else:
            self.scan_devices()

    def copy_text(self):
        text = self.text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.statusbar.showMessage("Copied!", 2000)

    def closeEvent(self, event):
        try:
            if self.worker:
                self.worker.stop()
                if self.thread:
                    self.thread.quit()
                    self.thread.wait(2000)
        except Exception as e:
            logger.exception("Close error")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = DictationApp()
    window.show()
    
    sys.exit(app.exec())