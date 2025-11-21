#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import time

# Check dependencies
try:
    import PyQt6
    import speech_recognition as sr
    import pyaudio
    import sounddevice as sd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install -r requirements.txt")
    sys.exit(1)

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QSize, QTimer, QSettings
from PyQt6.QtGui import QTextCursor, QFont, QPalette, QColor, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                            QComboBox, QSizePolicy, QStatusBar, QGroupBox,
                            QDialog, QMessageBox, QProgressBar, QListWidget, 
                            QListWidgetItem)
from PyQt6.QtCore import QPointF
from microphone_level_monitor import MicLevelBar

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
        
        while self._running:
            try:
                # Device-Info abrufen
                device_info = sd.query_devices(self.device_index)
                device_name = device_info['name'].lower()
                
                if not self.saved_config:
                    self.SAMPLERATE = device_info['default_samplerate']
                
                # Hardware-Warmup
                logger.info("⏳ Hardware-Warmup (300ms)...")
                self.status.emit("⏳ Mikrofon wird aktiviert...")
                time.sleep(0.3)
                
                # Stream erstellen
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
                
                # Stream-Warmup
                time.sleep(0.5)
                logger.info("Stream-Warmup abgeschlossen")
                
                # Warmup-Daten verwerfen
                self.recorded_data = []
                
                # Countdown
                for i in range(3, 0, -1):
                    self.status.emit(f"🔢 Countdown: {i}...")
                    time.sleep(1)
                
                self.countdown_done = True
                self.status.emit("🔴 AUFNAHME LÄUFT - SPRECHEN SIE JETZT!")
                logger.info("🔴 AUFNAHME GESTARTET")
                
                # Kontinuierliche Aufnahme-Schleife
                recording_duration = 3
                
                while self._running:
                    time.sleep(recording_duration)
                    
                    if not self.recorded_data:
                        continue
                    
                    # Audio-Daten zusammenführen
                    audio = np.concatenate(self.recorded_data) if self.recorded_data else np.array([])
                    self.recorded_data = []
                    
                    if len(audio) == 0:
                        continue
                    
                    # Send audio data to waveform widget
                    self.audioData.emit(audio)
                    
                    # Prüfe ob Audio laut genug ist
                    max_amp = np.max(np.abs(audio))
                    if max_amp < 0.001:
                        self.status.emit("🔴 AUFNAHME LÄUFT - Warte auf Sprache...")
                        continue
                    
                    # Konvertiere für Spracherkennung
                    self.status.emit("⚡ Verarbeite Audio...")
                    
                    audio_int16 = (audio * 32767).astype(np.int16)
                    if len(audio_int16.shape) > 1 and audio_int16.shape[1] > 1:
                        audio_int16 = np.mean(audio_int16, axis=1).astype(np.int16)
                    
                    audio_data = sr.AudioData(audio_int16.tobytes(), int(self.SAMPLERATE), 2)
                    
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
                                    continue
                            if not recognized_text:
                                self.status.emit("🔴 AUFNAHME LÄUFT - Sprechen Sie deutlicher...")
                        else:
                            text = recognizer.recognize_google(audio_data, language=self.language)
                            if text and text.strip():
                                logger.info(f"✅ Erkannt ({self.language}): {text}")
                                self.textReady.emit(text)
                                self.status.emit("🔴 AUFNAHME LÄUFT - Sprechen Sie weiter...")
                    except sr.UnknownValueError:
                        self.status.emit("🔴 AUFNAHME LÄUFT - Sprechen Sie deutlicher...")
                    except sr.RequestError as e:
                        self.error.emit(f"Netzwerkfehler: {e}")
                        time.sleep(2)

            except Exception as e:
                logger.error(f"Stream-Fehler: {e}")
                self.error.emit(f"Mikrofon-Fehler: {str(e)[:80]}")
                
                if self.stream:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                    self.stream = None
                
                if self._running:
                    time.sleep(2)
                    continue
                else:
                    break
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        logger.info("Worker beendet")
        self.finished.emit()

    def stop(self):
        self._running = False
        self.countdown_done = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass


class DeviceScannerWorker(QObject):
    """Worker for asynchronous device scanning"""
    deviceFound = pyqtSignal(dict)
    scanFinished = pyqtSignal(int, int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        self._running = True
        logger.info("Starting async device scan")
        
        working_count = 0
        failed_count = 0
        
        try:
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
                    default_input_index = None
            except:
                pass

            # Get all devices
            try:
                sd_devices = sd.query_devices()
            except Exception as e:
                self.error.emit(f"Error listing devices: {e}")
                self.scanFinished.emit(0, 0)
                self.finished.emit()
                return

            tested_devices = set()
            
            for idx, device in enumerate(sd_devices):
                if not self._running:
                    break
                    
                if device.get('max_input_channels', 0) > 0:
                    name = device.get('name', '')
                    lower = name.lower()
                    
                    # Filter out likely non-functional devices
                    if name == "Line ()" or (name.startswith("Input (") and "hands-free" not in lower):
                        continue
                        
                    self.status.emit(f"Testing: {name}...")
                    
                    # Test device (simplified test for speed)
                    try:
                        # Just check if we can open the stream with default settings
                        sr = int(device.get('default_samplerate', 44100))
                        try:
                            sd.check_input_settings(device=idx, channels=1, samplerate=sr)
                            
                            quality = 50
                            device_info = {
                                'idx': idx,
                                'name': name,
                                'lower': lower,
                                'sample_rate': sr,
                                'quality': quality,
                                'is_default': idx == default_input_index
                            }
                            
                            self.deviceFound.emit(device_info)
                            working_count += 1
                            tested_devices.add(name)
                            
                        except Exception as e:
                            failed_count += 1
                            
                    except Exception as e:
                        failed_count += 1
            
            # Fallback: SR devices
            try:
                sr_devices = sr.Microphone.list_microphone_names()
                for i, name in enumerate(sr_devices):
                    if not self._running:
                        break
                    if name not in tested_devices:
                        lower = (name or "").lower()
                        if any(x in lower for x in ['output', 'speaker', 'lautsprecher', 'spdif']):
                            continue
                            
                        self.deviceFound.emit({
                            'idx': i,
                            'name': f"[SR] {name}",
                            'lower': lower,
                            'sample_rate': 44100,
                            'quality': 40,
                            'is_default': False
                        })
                        working_count += 1
            except:
                pass

        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.error.emit(str(e))
            
        self.scanFinished.emit(working_count, failed_count)
        self.finished.emit()

    def stop(self):
        self._running = False


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
        
        # Start device monitoring (hot-plug detection)
        self.device_monitor = QTimer(self)
        self.device_monitor.timeout.connect(self.check_device_changes)
        self.device_monitor.start(3000)  # Check every 3 seconds
        
        # Aufnahme-Indicator ohne Blinken
        self.flash_timer = QTimer()
        self.flash_on = False
        
        # Initial scan
        QTimer.singleShot(100, self.start_async_scan)

    def setup_ui(self):
        """Setup the UI components"""
        # Main Layout
        main_layout = QVBoxLayout(self.central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Speech Recognition Pro")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2196F3;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        settings_btn = QPushButton("⚙️")
        settings_btn.setFixedSize(40, 40)
        settings_btn.clicked.connect(self.open_config_manager)
        header_layout.addWidget(settings_btn)
        
        main_layout.addLayout(header_layout)
        
        # Control Panel
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        # Mic Selection
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(250)
        self.mic_combo.currentIndexChanged.connect(self.on_device_changed)
        control_layout.addWidget(QLabel("Microphone:"))
        control_layout.addWidget(self.mic_combo)
        
        # Refresh Button
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setFixedSize(30, 30)
        self.refresh_btn.clicked.connect(self.start_async_scan)
        control_layout.addWidget(self.refresh_btn)
        
        control_layout.addSpacing(20)
        
        # Language Selection
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Auto-Detect", "German (DE)", "English (US)", "Russian (RU)", "French (FR)", "Spanish (ES)"])
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        control_layout.addWidget(QLabel("Language:"))
        control_layout.addWidget(self.lang_combo)
        
        control_layout.addStretch()
        
        # Action Buttons
        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.start_dictation)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px;")
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_dictation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px 16px;")
        control_layout.addWidget(self.stop_btn)
        
        self.test_btn = QPushButton("Test Mic")
        self.test_btn.clicked.connect(self.open_test_dialog)
        control_layout.addWidget(self.test_btn)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Visualization
        viz_layout = QHBoxLayout()
        
        # Waveform
        self.waveform = WaveformWidget()
        viz_layout.addWidget(self.waveform, stretch=2)
        
        # Level Bar
        self.mic_level_bar = MicLevelBar()
        viz_layout.addWidget(self.mic_level_bar, stretch=1)
        
        main_layout.addLayout(viz_layout)
        
        # Status Label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Recording Indicator
        self.recording_label = QLabel("🔴 RECORDING")
        self.recording_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_label.hide()
        main_layout.addWidget(self.recording_label)
        
        # Text Output
        self.text = QTextEdit()
        self.text.setPlaceholderText("Transcription will appear here...")
        self.text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        main_layout.addWidget(self.text)
        
        # Footer
        footer_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        footer_layout.addWidget(copy_btn)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)
        
        # Dark Theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QWidget { color: #ffffff; }
            QGroupBox { border: 1px solid #333; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QComboBox { background-color: #333; border: 1px solid #555; padding: 5px; border-radius: 4px; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #aaa; margin-right: 5px; }
            QPushButton { background-color: #333; border: 1px solid #555; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #444; }
            QPushButton:pressed { background-color: #222; }
            QTextEdit { background-color: #252526; border: 1px solid #333; border-radius: 4px; padding: 10px; }
            QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #2196F3; }
            QStatusBar { background-color: #007acc; color: white; }
        """)

    def start_async_scan(self):
        """Start asynchronous device scan"""
        self.mic_combo.clear()
        self.mic_combo.addItem("Scanning...", None)
        self.mic_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Scanning devices...")
        
        self.scan_worker = DeviceScannerWorker()
        self.scan_thread = QThread()
        self.scan_worker.moveToThread(self.scan_thread)
        
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.deviceFound.connect(self.on_device_found)
        self.scan_worker.scanFinished.connect(self.on_scan_finished)
        self.scan_worker.status.connect(self.on_scan_status)
        self.scan_worker.error.connect(self.on_scan_error)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        
        self.scan_thread.start()

    def on_device_found(self, device_info):
        """Handle found device"""
        if self.mic_combo.itemText(0) == "Scanning...":
            self.mic_combo.clear()
            
        name = device_info['name']
        idx = device_info['idx']
        self.mic_combo.addItem(f"{name}", idx)
        
        if device_info['is_default']:
            self.mic_combo.setCurrentIndex(self.mic_combo.count() - 1)

    def on_scan_finished(self, working, failed):
        """Handle scan completion"""
        self.mic_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"Scan complete. Found {working} devices.")
        self.update_start_button_state()

    def on_scan_status(self, msg):
        self.status_label.setText(msg)

    def on_scan_error(self, msg):
        logger.error(f"Scan error: {msg}")
        self.status_label.setText(f"Scan error: {msg}")

    def check_device_changes(self):
        """Check for device changes (hot-plug)"""
        try:
            # Quick check of device count
            devices = sd.query_devices()
            current_count = len(devices)
            
            # Also check names to detect swaps
            current_names = {d['name'] for d in devices}
            
            if current_count != self.last_device_count or current_names != self.last_device_names:
                logger.info("Device change detected")
                self.last_device_count = current_count
                self.last_device_names = current_names
                
                # Only trigger scan if not already scanning
                if self.refresh_btn.isEnabled():
                    self.start_async_scan()
        except:
            pass

    def on_device_changed(self, index):
        """Handle device selection change"""
        self.update_start_button_state()
        device_idx = self.mic_combo.currentData()
        if device_idx is not None:
            self.mic_level_bar.setDevice(device_idx)

    def on_language_changed(self, index):
        """Handle language selection change"""
        langs = ["auto", "de-DE", "en-US", "ru-RU", "fr-FR", "es-ES"]
        if 0 <= index < len(langs):
            self.language = langs[index]
        self.update_start_button_state()

    def load_saved_config_for_current_device(self):
        """Load saved config for currently selected device"""
        device_idx = self.mic_combo.currentData()
        if device_idx is None:
            return {}
            
        try:
            device_info = sd.query_devices(device_idx)
            device_name = device_info.get('name', '')
            key = self._device_config_key(device_name)
            
            config = {}
            if self.settings.contains(f"{key}_samplerate"):
                config['samplerate'] = self.settings.value(f"{key}_samplerate", type=int)
            if self.settings.contains(f"{key}_channels"):
                config['channels'] = self.settings.value(f"{key}_channels", type=int)
            if self.settings.contains(f"{key}_latency"):
                lat = self.settings.value(f"{key}_latency")
                config['latency'] = float(lat) if lat != "None" else None
            if self.settings.contains(f"{key}_host_api"):
                config['host_api'] = self.settings.value(f"{key}_host_api")
                
            return config
        except:
            return {}

    def _device_config_key(self, device_name):
        return f"device_{device_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')[:50]}"

    def open_test_dialog(self):
        """Open dialog to choose between single and batch test"""
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
            "• Your internet connection\n"
            "• Firewall settings\n"
            "• VPN configuration")

    def open_config_manager(self):
        """Open configuration manager dialog"""
        QMessageBox.information(self, "Settings", "Configuration manager not implemented yet.")

    def on_finished(self):
        """Handle worker finished"""
        logger.info("Worker finished")
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.worker = None
        self.thread = None
        
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
            self.start_async_scan()

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