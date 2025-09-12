# -*- coding: utf-8 -*-
import sys, os, importlib, subprocess, json, traceback, time
import numpy as np
import logging
from datetime import datetime
import glob

# Clean up old log files
for old_log in glob.glob("mikrophon_test_*.log"):
    try:
        os.remove(old_log)
    except:
        pass  # Ignore errors if file is in use

# Configure logging with single file
log_filename = "mikrophon_test.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),  # 'w' mode overwrites
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure(pkg):
    try:
        return importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
        return importlib.import_module(pkg)

def ensure_package(pkg_name, import_name=None):
    """Ensure a package is installed, with optional different import name"""
    if import_name is None:
        import_name = pkg_name
    try:
        return importlib.import_module(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name, "--quiet"])
        return importlib.import_module(import_name)

QtCore = ensure("PyQt6.QtCore")
QtGui = ensure("PyQt6.QtGui")
QtWidgets = ensure("PyQt6.QtWidgets")
sd = ensure("sounddevice")

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings, QPointF
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QColor, QAction, QPalette, QIcon, QPainter, QPen, QBrush
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableView, QLabel, QHeaderView, QSizePolicy, QMenu, QMessageBox, QDialog, QProgressBar, QTextEdit, QLineEdit, QCheckBox, QComboBox, QFileDialog

class DeviceCollector(QThread):
    resultReady = pyqtSignal(list, str)
    error = pyqtSignal(str)
    def run(self):
        try:
            # Cache leeren und Geräte neu initialisieren
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            hostapi_names = {i: h.get("name", str(i)) for i, h in enumerate(hostapis)}
            default_in = None
            try:
                din, dout = sd.default.device
                default_in = din
            except Exception:
                default_in = None
            out = []
            for idx, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    test_ok = True
                    try:
                        sr = d.get("default_samplerate", None)
                        sr = int(sr) if sr else None
                        ch = 1 if d.get("max_input_channels", 0) >= 1 else d.get("max_input_channels", 0)
                        sd.check_input_settings(device=idx, channels=ch, samplerate=sr)
                    except Exception:
                        test_ok = False
                    is_default = False
                    if isinstance(default_in, int):
                        is_default = (idx == default_in)
                    elif isinstance(default_in, str):
                        is_default = (d.get("name") == default_in)
                    record = {
                        "Index": idx,
                        "Name": d.get("name", ""),
                        "Host API": hostapi_names.get(d.get("hostapi"), str(d.get("hostapi"))),
                        "Max Input Channels": d.get("max_input_channels", 0),
                        "Default Samplerate": d.get("default_samplerate", None),
                        "Low Input Latency": d.get("default_low_input_latency", None),
                        "High Input Latency": d.get("default_high_input_latency", None),
                        "Ist Standard-Eingabe": is_default,
                        "Nutzbar (Test)": test_ok,
                        "Rohdaten": d
                    }
                    out.append(record)
            self.resultReady.emit(out, "")
        except Exception as e:
            self.error.emit("Fehler bei der Gerätesuche:\n" + "".join(traceback.format_exception(e)))

class WaveformWidget(QWidget):
    """Widget to display audio waveform in real-time"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(150)
        self.audio_data = []
        self.max_points = 500  # Number of points to display
        
    def update_waveform(self, audio_chunk):
        """Update waveform with new audio data"""
        if audio_chunk is None or len(audio_chunk) == 0:
            return
            
        # Downsample for display
        step = max(1, len(audio_chunk) // 10)
        downsampled = audio_chunk[::step]
        
        # Add to buffer
        self.audio_data.extend(downsampled.flatten().tolist())
        
        # Keep only last max_points
        if len(self.audio_data) > self.max_points:
            self.audio_data = self.audio_data[-self.max_points:]
        
        self.update()  # Trigger repaint
    
    def clear_waveform(self):
        """Clear the waveform display"""
        self.audio_data = []
        self.update()
    
    def paintEvent(self, event):
        """Paint the waveform"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if not self.audio_data:
            # Draw center line when no data
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)
            return
        
        # Draw waveform
        painter.setPen(QPen(QColor(0, 255, 100), 2))
        
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # Scale audio data to widget height
        max_val = max(abs(min(self.audio_data, default=0)), abs(max(self.audio_data, default=0)))
        if max_val == 0:
            max_val = 1
        
        points = []
        for i, value in enumerate(self.audio_data):
            x = (i / len(self.audio_data)) * width
            y = center_y - (value / max_val) * (height // 2 - 5)
            points.append(QPointF(x, y))
        
        # Draw the waveform
        for i in range(1, len(points)):
            painter.drawLine(points[i-1], points[i])
        
        # Draw center line
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(0, center_y, width, center_y)

class MicrophoneTestDialog(QDialog):
    def __init__(self, device_idx, device_name, parent=None):
        super().__init__(parent)
        self.device_idx = device_idx
        self.device_name = device_name
        self.is_recording = False
        self.recorded_data = []
        self.working_params = {}  # Store working parameters
        self.countdown_timer = QTimer()
        self.countdown_value = 3
        self.recording_duration = 10  # Extended to 10 seconds for better speech capture
        self.test_number = 0  # Track test number
        self.last_recording = None  # Store last recording for playback
        self.last_samplerate = 44100  # Store samplerate for playback
        
        # Load saved configuration for this device
        self.settings = QSettings('MikrofoneTool', 'DeviceConfigs')
        self.config_key = f"device_{device_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')[:50]}"
        self.saved_config = None
        self.load_saved_config()
        
        logger.info(f"=== MicrophoneTestDialog initialized for {device_name} (idx: {device_idx}) ===")
        
        # Try to import speech_recognition for text transcription
        try:
            self.sr = ensure_package('SpeechRecognition', 'speech_recognition')
            self.speech_recognition_available = True
        except:
            self.sr = None
            self.speech_recognition_available = False
        
        self.setWindowTitle(f"Mikrofon Test: {device_name}")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout()
        
        info_label = QLabel(f"Gerät: {device_name} (Index: {device_idx})")
        layout.addWidget(info_label)
        
        self.status_label = QLabel("Drücken Sie 'Test starten' und sprechen Sie etwas...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        self.level_bar = QProgressBar()
        self.level_bar.setMaximum(100)
        self.level_bar.setTextVisible(True)
        self.level_bar.setFormat("Pegel: %v%")
        layout.addWidget(self.level_bar)
        
        # Add waveform display
        waveform_label = QLabel("🎵 Audio-Wellenform:")
        layout.addWidget(waveform_label)
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Testergebnisse erscheinen hier...")
        layout.addWidget(self.result_text)
        
        btn_layout = QHBoxLayout()
        self.test_btn = QPushButton("Test starten")
        self.test_btn.clicked.connect(self.toggle_test)
        btn_layout.addWidget(self.test_btn)
        
        self.play_btn = QPushButton("🔊 Anhören")
        self.play_btn.clicked.connect(self.play_recording)
        self.play_btn.setEnabled(False)  # Initially disabled
        self.play_btn.setToolTip("Letzte Aufnahme abspielen")
        btn_layout.addWidget(self.play_btn)
        
        self.save_btn = QPushButton("💾 Speichern")
        self.save_btn.clicked.connect(self.save_recording)
        self.save_btn.setEnabled(False)  # Initially disabled
        self.save_btn.setToolTip("Aufnahme als WAV speichern")
        btn_layout.addWidget(self.save_btn)
        
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_level)
        
    def toggle_test(self):
        if not self.is_recording:
            self.test_number += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST #{self.test_number} STARTED - Device: {self.device_name} (idx: {self.device_idx})")
            logger.info(f"{'='*60}")
            self.start_recording()
        else:
            logger.info(f"Stopping test #{self.test_number}")
            self.stop_recording()
    
    def start_recording(self):
        try:
            # Clean up any existing stream first
            if hasattr(self, 'stream') and self.stream:
                logger.info("Cleaning up existing stream")
                try:
                    self.stream.stop()
                    self.stream.close()
                    logger.info("Existing stream closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing stream: {e}")
                self.stream = None
            
            # Clear any old data
            prev_data_len = len(self.recorded_data)
            self.recorded_data = []
            logger.info(f"Cleared recorded_data (had {prev_data_len} chunks)")
            
            device = sd.query_devices(self.device_idx)
            samplerate = int(device['default_samplerate'])
            max_channels = device.get('max_input_channels', 1)
            channels = min(1, max_channels)  # Use mono if available
            host_api_name = sd.query_hostapis()[device['hostapi']]['name']
            
            self.result_text.clear()
            self.result_text.append(f"Test gestartet...\n")
            self.result_text.append(f"Gerät: {device.get('name', 'Unknown')}")
            self.result_text.append(f"Host API: {host_api_name}")
            self.result_text.append(f"Max. Eingabekanäle: {max_channels}")
            self.result_text.append(f"Samplerate: {samplerate} Hz")
            self.result_text.append(f"Verwende Kanäle: {channels}\n")
            
            # Get device latency settings
            low_latency = device.get('default_low_input_latency', 0.01)
            high_latency = device.get('default_high_input_latency', 0.1)
            
            # Try different configurations if the default fails
            # Prioritize standard configurations that work with modern Bluetooth devices
            configurations = []
            
            # Add saved configuration first if available
            if self.saved_config:
                saved_ch = self.saved_config['channels']
                saved_sr = self.saved_config['samplerate']
                saved_lat = self.saved_config['latency']
                configurations.append((saved_ch, saved_sr, saved_lat))
                self.result_text.append(f"💾 Verwende gespeicherte Konfiguration: {saved_sr} Hz, {saved_ch} Kanal(e)\n")
            
            # Add standard configurations
            configurations.extend([
                (channels, 44100, None),  # Standard rate, works with modern Bluetooth (e.g., Buds3 Pro)
                (channels, samplerate, None),  # Default latency
                (channels, 44100, high_latency),  # 44100 Hz with high latency for MME
                (channels, samplerate, high_latency),  # High latency for MME
                (channels, 48000, high_latency),
                (channels, 16000, high_latency),  # Legacy Bluetooth devices (WDM-KS)
                (channels, 8000, high_latency),
                (max_channels, samplerate, high_latency),  # Try with max channels
                (max_channels, 44100, high_latency),
            ])
            
            stream_created = False
            last_error = None
            
            logger.info(f"Trying {len(configurations)} configurations...")
            for config_idx, (ch, sr, latency) in enumerate(configurations, 1):
                try:
                    config_str = f"Config #{config_idx}: {sr} Hz, {ch} ch, latency={latency}"
                    logger.info(f"Trying {config_str}")
                    self.result_text.append(f"Teste: {sr} Hz, {ch} Kanal(e), Latenz: {latency}")
                    
                    # Test configuration first
                    sd.check_input_settings(device=self.device_idx, channels=ch, samplerate=sr)
                    logger.info(f"  Configuration check passed")
                    
                    self.is_recording = True
                    self.recorded_data = []
                    self.test_btn.setText("Test stoppen")
                    self.status_label.setText("Aufnahme läuft... Sprechen Sie jetzt!")
                    
                    # Prepare stream parameters
                    stream_params = {
                        'device': self.device_idx,
                        'channels': ch,
                        'samplerate': sr,
                        'dtype': 'float32',
                        'callback': self.audio_callback,
                        'blocksize': 512  # Smaller blocksize for better compatibility
                    }
                    
                    # Add latency if specified
                    if latency is not None:
                        stream_params['latency'] = latency
                    
                    # Start audio stream with tested configuration
                    logger.info(f"  Creating stream with params: {stream_params}")
                    self.stream = sd.InputStream(**stream_params)
                    self.stream.start()
                    logger.info(f"  Stream started successfully")
                    self.timer.start(100)  # Update every 100ms
                    
                    # Store working parameters for later use
                    self.working_params = {
                        'device': self.device_idx,
                        'samplerate': sr,
                        'channels': ch,
                        'blocksize': 512,
                        'latency': latency,
                        'dtype': 'float32',
                        'host_api': host_api_name
                    }
                    
                    # Save successful configuration
                    if hasattr(self, 'settings'):
                        self.save_config(sr, ch, latency)
                    
                    self.result_text.append(f"✓ ERFOLG mit {sr} Hz, {ch} Kanal(e), Latenz: {latency}")
                    
                    # Give microphone time to warm up
                    import time
                    time.sleep(0.3)  # 300ms warmup (increased)
                    
                    # Clear any initial noise
                    prev_len = len(self.recorded_data)
                    self.recorded_data = []
                    logger.info(f"  Cleared initial noise ({prev_len} chunks)")
                    
                    stream_created = True
                    logger.info(f"  SUCCESS: Stream created with config #{config_idx}")
                    break
                    
                except Exception as config_error:
                    last_error = str(config_error)
                    self.result_text.append(f"× Fehlgeschlagen: {last_error[:100]}")
                    continue
            
            if stream_created:
                # Add warmup delay to let microphone activate properly
                self.result_text.append(f"\n⏳ Mikrofon wird aktiviert...")
                # Clear any initial noise during warmup
                prev_len = len(self.recorded_data)
                self.recorded_data = []
                logger.info(f"Cleared warmup noise ({prev_len} chunks)")
                QTimer.singleShot(500, self.start_countdown)  # 500ms warmup
            else:
                # Try alternative approach: recording without callback
                self.try_blocking_recording()
            
        except Exception as e:
            self.result_text.append(f"\nFEHLER beim Starten: {str(e)}")
            self.result_text.append(f"\nBitte prüfen Sie:")
            self.result_text.append(f"- Ist das Mikrofon richtig angeschlossen?")
            self.result_text.append(f"- Sind die Windows-Audioeinstellungen korrekt?")
            self.result_text.append(f"- Hat eine andere Anwendung das Mikrofon blockiert?")
            self.is_recording = False
            self.test_btn.setText("Test starten")
    
    def try_blocking_recording(self):
        """Fallback: Try blocking recording without callback"""
        try:
            self.result_text.append(f"\nVersuche alternative Aufnahmemethode...")
            device = sd.query_devices(self.device_idx)
            
            # Try to find alternative device with same name in different host API
            all_devices = sd.query_devices()
            device_name = device['name']
            
            # Look for same device in Windows DirectSound or WASAPI
            alternative_device = None
            for idx, dev in enumerate(all_devices):
                if (dev.get('max_input_channels', 0) > 0 and 
                    device_name in dev['name'] and 
                    idx != self.device_idx):
                    host_api = sd.query_hostapis()[dev['hostapi']]['name']
                    if 'DirectSound' in host_api or 'WASAPI' in host_api:
                        alternative_device = idx
                        self.result_text.append(f"Gefunden: Alternative API - {host_api} (Index: {idx})")
                        break
            
            # Try alternative device first if found
            if alternative_device is not None:
                try:
                    self.result_text.append(f"Teste alternatives Gerät (Index {alternative_device})...")
                    
                    # Get alternative device info
                    alt_device = sd.query_devices(alternative_device)
                    alt_samplerate = int(alt_device.get('default_samplerate', 44100))
                    
                    self.result_text.append(f"Alternative Samplerate: {alt_samplerate} Hz")
                    
                    duration = 8  # Increased duration for speech recognition
                    channels = 1
                    
                    # Add warmup info
                    self.result_text.append(f"Warmup für Mikrofon-Aktivierung...")
                    import time
                    time.sleep(0.5)  # 500ms warmup
                    
                    # Try with alternative device's default samplerate first
                    for test_sr in [alt_samplerate, 48000, 44100, 16000, 8000]:
                        try:
                            self.result_text.append(f"Versuche {test_sr} Hz...")
                            recording = sd.rec(int(duration * test_sr),
                                             samplerate=test_sr,
                                             channels=channels,
                                             device=alternative_device,
                                             dtype='float32')
                            sd.wait()
                            
                            max_val = np.max(np.abs(recording))
                            self.result_text.append(f"\n✅ ERFOLG mit alternativer API!")
                            self.result_text.append(f"Samplerate: {test_sr} Hz")
                            self.result_text.append(f"Max. Amplitude: {max_val:.4f}")
                            
                            if max_val > 0.001:
                                self.result_text.append(f"Audio-Signal erkannt!")
                                # Try speech recognition if available
                                self.try_speech_recognition(recording, test_sr)
                                
                                # Log all working parameters
                                self.result_text.append(f"\n=== ERFOLGREICHE PARAMETER (Alternative API) ===")
                                self.result_text.append(f"Device Index: {alternative_device}")
                                self.result_text.append(f"Host API: {host_api}")
                                self.result_text.append(f"Samplerate: {test_sr} Hz")
                                self.result_text.append(f"Channels: {channels}")
                                self.result_text.append(f"Aufnahmedauer: {duration}s")
                                self.result_text.append(f"Dtype: float32")
                            else:
                                self.result_text.append(f"⚠️ Sehr leises Signal")
                            
                            # Store working parameters
                            self.working_params = {
                                'device': alternative_device,
                                'samplerate': test_sr,
                                'channels': channels,
                                'blocksize': 512,
                                'latency': None,
                                'dtype': 'float32',
                                'host_api': host_api,
                                'duration': duration
                            }
                            
                            self.is_recording = False
                            self.test_btn.setText("Test starten")
                            self.test_btn.setEnabled(True)
                            return
                            
                        except Exception as sr_error:
                            self.result_text.append(f"  Fehler bei {test_sr} Hz")
                            continue
                            
                except Exception as alt_error:
                    self.result_text.append(f"Alternative fehlgeschlagen: {str(alt_error)[:100]}")
            
            # Fallback to original device with blocking
            duration = 8  # seconds (increased for better speech capture)
            samplerate = 44100
            channels = device.get('max_input_channels', 1)
            
            self.result_text.append(f"Blockierende Aufnahme für {duration} Sekunden...")
            self.result_text.append(f"⏳ Mikrofon wird aktiviert (1 Sekunde Warmup)...")
            self.is_recording = True
            self.test_btn.setText("Aufnahme läuft...")
            self.test_btn.setEnabled(False)
            
            # Warmup phase - let mic activate
            import time
            time.sleep(1.0)  # 1 second warmup
            self.result_text.append(f"🔴 AUFNAHME STARTET JETZT!")
            
            # Record audio
            recording = sd.rec(int(duration * samplerate), 
                             samplerate=samplerate, 
                             channels=channels,
                             device=self.device_idx,
                             dtype='float32')
            sd.wait()  # Wait until recording is finished
            
            self.result_text.append(f"\n=== TESTERGEBNISSE (Blockierend) ===")
            self.result_text.append(f"Aufnahmedauer: {duration} Sekunden")
            self.result_text.append(f"Samples: {len(recording)}")
            
            max_val = np.max(np.abs(recording))
            avg_val = np.mean(np.abs(recording))
            
            self.result_text.append(f"Max. Amplitude: {max_val:.4f}")
            self.result_text.append(f"Durchschn. Amplitude: {avg_val:.6f}")
            
            if max_val > 0.001:
                self.result_text.append(f"\n✅ Mikrofon funktioniert (blockierende Methode)")
                # Try speech recognition if available
                self.try_speech_recognition(recording, samplerate)
                
                # Log all working parameters
                host_api_blocking = sd.query_hostapis()[device['hostapi']]['name']
                self.result_text.append(f"\n=== ERFOLGREICHE PARAMETER (Blockierend) ===")
                self.result_text.append(f"Device Index: {self.device_idx}")
                self.result_text.append(f"Host API: {host_api_blocking}")
                self.result_text.append(f"Samplerate: {samplerate} Hz")
                self.result_text.append(f"Channels: {channels}")
                self.result_text.append(f"Aufnahmedauer: {duration}s")
                self.result_text.append(f"Warmup verwendet: 1s")
                self.result_text.append(f"Dtype: float32")
                
                # Store working parameters
                self.working_params = {
                    'device': self.device_idx,
                    'samplerate': samplerate,
                    'channels': channels,
                    'blocksize': 512,
                    'latency': None,
                    'dtype': 'float32',
                    'host_api': host_api_blocking,
                    'duration': duration,
                    'warmup': 1.0
                }
            else:
                self.result_text.append(f"\n⚠️ Kein oder sehr leises Signal")
                
        except Exception as e:
            self.result_text.append(f"\nBlockierende Aufnahme fehlgeschlagen: {str(e)}")
        finally:
            self.is_recording = False
            self.test_btn.setText("Test starten")
            self.test_btn.setEnabled(True)
    
    def try_speech_recognition(self, audio_data, samplerate):
        """Try to recognize speech from the recorded audio"""
        if not self.speech_recognition_available or self.sr is None:
            return
        
        try:
            self.result_text.append(f"\n=== SPRACHERKENNUNG ===")
            self.result_text.append(f"Versuche gesprochenen Text zu erkennen...")
            
            # Convert float32 to int16 for speech recognition
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # If stereo, convert to mono
            if len(audio_int16.shape) > 1 and audio_int16.shape[1] > 1:
                audio_int16 = np.mean(audio_int16, axis=1).astype(np.int16)
            
            # Create AudioData object
            audio = self.sr.AudioData(
                audio_int16.tobytes(),
                int(samplerate),
                2  # 2 bytes per sample (int16)
            )
            
            # Create recognizer
            r = self.sr.Recognizer()
            
            # Try German first, then English
            try:
                text_de = r.recognize_google(audio, language="de-DE")
                self.result_text.append(f"📝 Erkannter Text (Deutsch): \"{text_de}\"")
            except:
                try:
                    text_en = r.recognize_google(audio, language="en-US")
                    self.result_text.append(f"📝 Recognized text (English): \"{text_en}\"")
                except:
                    self.result_text.append(f"⚠️ Konnte keinen Text erkennen (zu leise oder undeutlich)")
                    
        except Exception as e:
            self.result_text.append(f"Spracherkennung fehlgeschlagen: {str(e)[:100]}")
    
    def start_countdown(self):
        """Start countdown before recording"""
        # Clear any warmup data before countdown
        prev_len = len(self.recorded_data)
        self.recorded_data = []
        logger.info(f"\nStarting countdown - cleared {prev_len} warmup chunks")
        
        self.countdown_value = 3
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)  # Update every second
        self.update_countdown()
    
    def update_countdown(self):
        """Update countdown display"""
        if self.countdown_value > 0:
            self.status_label.setText(f"🎤 Aufnahme startet in {self.countdown_value}... Bereiten Sie sich vor!")
            self.result_text.append(f"Countdown: {self.countdown_value}...")
            logger.info(f"Countdown: {self.countdown_value}")
            QApplication.processEvents()  # Update UI immediately
            self.countdown_value -= 1
        else:
            self.countdown_timer.stop()
            logger.info(f"\n*** RECORDING STARTED for {self.recording_duration}s ***")
            logger.info(f"Current data buffer: {len(self.recorded_data)} chunks")
            self.recorded_data = []
            logger.info(f"Cleared countdown data before actual recording")
            
            # Clear waveform for fresh display
            if hasattr(self, 'waveform'):
                self.waveform.clear_waveform()
            
            self.status_label.setText(f"🔴 AUFNAHME LÄUFT für {self.recording_duration} Sekunden! Sprechen Sie jetzt deutlich!")
            self.result_text.append(f"\n🔴 AUFNAHME GESTARTET - {self.recording_duration} Sekunden Aufnahmezeit")
            self.result_text.append(f"Sprechen Sie jetzt einen vollständigen Satz!")
            QApplication.processEvents()  # Update UI immediately
            
            # DO NOT clear recorded data here - we want to keep recording!
            # The stream is already running and collecting data
            
            # Small delay to ensure clean start
            QTimer.singleShot(100, lambda: self.result_text.append("Audio-Aufnahme aktiv..."))
            
            # Start actual recording and auto-stop
            QTimer.singleShot(self.recording_duration * 1000, self.stop_recording)
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Always record data - don't filter based on volume
        # The user might speak quietly or there might be a delay
        if indata is not None and len(indata) > 0:
            self.recorded_data.append(indata.copy())
            
            # Update waveform display in UI thread
            if self.is_recording and hasattr(self, 'waveform'):
                # Use QTimer to update in main thread
                try:
                    QTimer.singleShot(0, lambda data=indata.copy(): self.waveform.update_waveform(data))
                except:
                    pass  # Ignore errors in waveform update
    
    def update_level(self):
        if self.recorded_data:
            # Get last chunk
            last_chunk = self.recorded_data[-1]
            # Calculate RMS level
            rms = np.sqrt(np.mean(last_chunk**2))
            # Convert to percentage (0-100)
            level = min(100, int(rms * 500))
            self.level_bar.setValue(level)
            
            # Also update waveform with last chunk
            if hasattr(self, 'waveform') and self.is_recording:
                self.waveform.update_waveform(last_chunk)
    
    def stop_recording(self):
        if not self.is_recording:
            logger.warning("stop_recording called but not recording")
            return
            
        logger.info(f"\nStopping recording - collected {len(self.recorded_data)} chunks")
        self.is_recording = False
        self.timer.stop()
        
        # Store stream parameters before closing
        stream_samplerate = None
        stream_channels = None
        stream_blocksize = None
        stream_latency = None
        
        if hasattr(self, 'stream') and self.stream:
            try:
                stream_samplerate = self.stream.samplerate
                stream_channels = self.stream.channels
                stream_blocksize = self.stream.blocksize
                stream_latency = self.stream.latency
            except:
                # Use defaults if stream attributes are not accessible
                stream_samplerate = 44100
                stream_channels = 1
        
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None  # Clear stream reference
        except Exception as e:
            self.result_text.append(f"Stream-Stop Warnung: {str(e)[:50]}")
        
        self.test_btn.setText("Test starten")
        self.status_label.setText(f"✅ Test abgeschlossen - {len(self.recorded_data)} Chunks aufgenommen")
        self.level_bar.setValue(0)
        
        # Stop countdown timer if still running
        if hasattr(self, 'countdown_timer'):
            self.countdown_timer.stop()
        
        # Debug: Show how much data was recorded
        self.result_text.append(f"\nDebug: {len(self.recorded_data)} Audio-Chunks aufgenommen")
        
        # Analyze recorded data
        if self.recorded_data and len(self.recorded_data) > 0:
            all_data = np.concatenate(self.recorded_data)
            # Use stored samplerate or fallback to default
            samplerate = stream_samplerate if stream_samplerate else 44100
            
            # Store for playback
            self.last_recording = all_data
            self.last_samplerate = samplerate
            self.play_btn.setEnabled(True)  # Enable playback button
            self.save_btn.setEnabled(True)  # Enable save button
            duration = len(all_data) / samplerate
            max_val = np.max(np.abs(all_data))
            avg_val = np.mean(np.abs(all_data))
            
            self.result_text.append(f"\n=== TESTERGEBNISSE ===")
            self.result_text.append(f"Aufnahmedauer: {duration:.2f} Sekunden")
            self.result_text.append(f"Samples aufgenommen: {len(all_data)}")
            self.result_text.append(f"Max. Amplitude: {max_val:.4f}")
            self.result_text.append(f"Durchschn. Amplitude: {avg_val:.6f}")
            
            if max_val > 0.01:
                self.result_text.append(f"\n✅ ERFOLG: Mikrofon funktioniert!")
                self.result_text.append(f"Audio-Signal wurde erkannt.")
                
                # Try speech recognition on recorded data
                self.try_speech_recognition(all_data, samplerate)
                
                # Ausgabe der exakten Parameter
                device = sd.query_devices(self.device_idx)
                host_api = sd.query_hostapis()[device['hostapi']]['name']
                
                # Get additional device information
                low_latency = device.get('default_low_input_latency', None)
                high_latency = device.get('default_high_input_latency', None)
                
                self.result_text.append(f"\n=== FUNKTIONIERENDE KONFIGURATION ===")
                self.result_text.append(f"Device Index: {self.device_idx}")
                self.result_text.append(f"Device Name: {device['name']}")
                self.result_text.append(f"Host API: {host_api}")
                self.result_text.append(f"Samplerate: {stream_samplerate if stream_samplerate else samplerate} Hz")
                self.result_text.append(f"Channels: {stream_channels if stream_channels else 1}")
                self.result_text.append(f"Blocksize: {stream_blocksize if stream_blocksize else 512}")
                self.result_text.append(f"Latency (aktuell): {stream_latency:.6f}s" if stream_latency else "Latency: N/A")
                self.result_text.append(f"Low Input Latency: {low_latency:.6f}s" if low_latency else "Low Input Latency: N/A")
                self.result_text.append(f"High Input Latency: {high_latency:.6f}s" if high_latency else "High Input Latency: N/A")
                self.result_text.append(f"Dtype: float32")
                self.result_text.append(f"")
                self.result_text.append(f"=== KRITISCHE TIMING-PARAMETER ===")
                self.result_text.append(f"Warmup vor Stream: 300ms (Hardware-Aktivierung)")
                self.result_text.append(f"Warmup nach Stream: 500ms (Stream-Stabilisierung)")
                self.result_text.append(f"Countdown: 3 Sekunden (Vorbereitung)")
                self.result_text.append(f"Aufnahmedauer: {self.recording_duration}s (tatsächliche Aufnahme)")
                self.result_text.append(f"WICHTIG: Daten nur VOR Countdown leeren, NIE danach!")
                
                self.result_text.append(f"\n=== PYTHON CODE FÜR ANDERE APPS ===")
                self.result_text.append(f"```python")
                self.result_text.append(f"import sounddevice as sd")
                self.result_text.append(f"import numpy as np")
                self.result_text.append(f"")
                self.result_text.append(f"# Exakte Parameter für dieses Mikrofon")
                self.result_text.append(f"DEVICE_INDEX = {self.device_idx}")
                self.result_text.append(f"SAMPLERATE = {stream_samplerate if stream_samplerate else samplerate}")
                self.result_text.append(f"CHANNELS = {stream_channels if stream_channels else 1}")
                self.result_text.append(f"BLOCKSIZE = {stream_blocksize if stream_blocksize else 512}")
                self.result_text.append(f"DTYPE = 'float32'")
                self.result_text.append(f"LATENCY = {stream_latency:.6f}  # Sekunden" if stream_latency else "LATENCY = None  # Default")
                self.result_text.append(f"HOST_API = '{host_api}'  # Wichtig für Bluetooth!")
                self.result_text.append(f"")
                self.result_text.append(f"# Option 1: Stream mit Callback (EMPFOHLEN für Echtzeit)")
                self.result_text.append(f"recorded_data = []  # WICHTIG: Global für Callback")
                self.result_text.append(f"")
                self.result_text.append(f"def audio_callback(indata, frames, time, status):")
                self.result_text.append(f"    if status:")
                self.result_text.append(f"        print(f'Callback Status: {{status}}')")
                self.result_text.append(f"    # WICHTIG: IMMER alle Daten aufnehmen, nicht filtern!")
                self.result_text.append(f"    if indata is not None:")
                self.result_text.append(f"        recorded_data.append(indata.copy())")
                self.result_text.append(f"")
                self.result_text.append(f"# Stream erstellen und starten")
                self.result_text.append(f"stream = sd.InputStream(")
                self.result_text.append(f"    device=DEVICE_INDEX,")
                self.result_text.append(f"    channels=CHANNELS,")
                self.result_text.append(f"    samplerate=SAMPLERATE,")
                self.result_text.append(f"    blocksize=BLOCKSIZE,")
                self.result_text.append(f"    latency=LATENCY,  # Wichtig für Timing!")
                self.result_text.append(f"    callback=audio_callback)")
                self.result_text.append(f"")
                self.result_text.append(f"with stream:")
                self.result_text.append(f"    time.sleep(0.5)  # KRITISCH: Warmup für Stream")
                self.result_text.append(f"    recorded_data = []  # Warmup-Daten verwerfen")
                self.result_text.append(f"    print('3...')")
                self.result_text.append(f"    time.sleep(1)")
                self.result_text.append(f"    print('2...')")
                self.result_text.append(f"    time.sleep(1)")
                self.result_text.append(f"    print('1...')")
                self.result_text.append(f"    time.sleep(1)")
                self.result_text.append(f"    print('🔴 AUFNAHME!')")
                self.result_text.append(f"    # NICHT recorded_data leeren hier!")
                self.result_text.append(f"    time.sleep(10)  # 10 Sekunden aufnehmen")
                self.result_text.append(f"")
                self.result_text.append(f"# Aufgenommene Daten verarbeiten")
                self.result_text.append(f"if recorded_data:")
                self.result_text.append(f"    audio = np.concatenate(recorded_data)")
                self.result_text.append(f"    print(f'Aufgenommen: {{len(audio)/SAMPLERATE:.2f}}s')")
                self.result_text.append(f"    print(f'Max Amplitude: {{np.max(np.abs(audio)):.4f}}')")
                self.result_text.append(f"")
                self.result_text.append(f"# Option 2: Blockierende Aufnahme (einfacher aber weniger Kontrolle)")
                self.result_text.append(f"import time")
                self.result_text.append(f"time.sleep(0.5)  # KRITISCH: Hardware-Warmup")
                self.result_text.append(f"print('Countdown: 3-2-1...')")
                self.result_text.append(f"time.sleep(3)")
                self.result_text.append(f"print('🔴 AUFNAHME!')")
                self.result_text.append(f"duration = 10  # Sekunden")
                self.result_text.append(f"recording = sd.rec(")
                self.result_text.append(f"    int(duration * SAMPLERATE),")
                self.result_text.append(f"    samplerate=SAMPLERATE,")
                self.result_text.append(f"    channels=CHANNELS,")
                self.result_text.append(f"    device=DEVICE_INDEX,")
                self.result_text.append(f"    dtype=DTYPE,")
                self.result_text.append(f"    latency=LATENCY)")
                self.result_text.append(f"sd.wait()")
                self.result_text.append(f"print(f'Max: {{np.max(np.abs(recording)):.4f}}')")
                self.result_text.append(f"")
                self.result_text.append(f"# Option 3: Mit Spracherkennung (BESTE LÖSUNG)")
                self.result_text.append(f"import speech_recognition as sr")
                self.result_text.append(f"import time")
                self.result_text.append(f"")
                self.result_text.append(f"# KRITISCHE ERKENNTNISSE:")
                self.result_text.append(f"# 1. Warmup VOR Stream-Start (300ms Hardware)")
                self.result_text.append(f"# 2. Warmup NACH Stream-Start (500ms Stabilisierung)")
                self.result_text.append(f"# 3. Daten NUR vor Countdown leeren, NIE danach!")
                self.result_text.append(f"# 4. ALLE Daten aufnehmen (nicht nach Lautstärke filtern)")
                self.result_text.append(f"")
                self.result_text.append(f"print('Mikrofon wird aktiviert...')")
                self.result_text.append(f"time.sleep(0.5)  # Hardware-Warmup")
                self.result_text.append(f"")
                self.result_text.append(f"# Stream-basierte Aufnahme mit Timing-Kontrolle")
                self.result_text.append(f"recorded_data = []")
                self.result_text.append(f"")
                self.result_text.append(f"def callback(indata, frames, time, status):")
                self.result_text.append(f"    recorded_data.append(indata.copy())")
                self.result_text.append(f"")
                self.result_text.append(f"stream = sd.InputStream(device=DEVICE_INDEX, channels=CHANNELS,")
                self.result_text.append(f"                        samplerate=SAMPLERATE, blocksize=BLOCKSIZE,")
                self.result_text.append(f"                        dtype=DTYPE, latency=LATENCY, callback=callback)")
                self.result_text.append(f"")
                self.result_text.append(f"with stream:")
                self.result_text.append(f"    time.sleep(0.5)  # Stream-Warmup")
                self.result_text.append(f"    recorded_data = []  # Warmup-Daten löschen")
                self.result_text.append(f"    ")
                self.result_text.append(f"    # Countdown")
                self.result_text.append(f"    for i in range(3, 0, -1):")
                self.result_text.append(f"        print(f'{{i}}...')")
                self.result_text.append(f"        time.sleep(1)")
                self.result_text.append(f"    ")
                self.result_text.append(f"    print('🔴 AUFNAHME LÄUFT! Sprechen Sie jetzt!')")
                self.result_text.append(f"    # WICHTIG: recorded_data NICHT leeren!")
                self.result_text.append(f"    time.sleep(10)  # 10 Sekunden aufnehmen")
                self.result_text.append(f"")
                self.result_text.append(f"# Daten zusammenführen")
                self.result_text.append(f"audio = np.concatenate(recorded_data) if recorded_data else np.array([])")
                self.result_text.append(f"print(f'Aufgenommen: {{len(audio)/SAMPLERATE:.2f}}s, Max: {{np.max(np.abs(audio)):.4f}}')")
                self.result_text.append(f"")
                self.result_text.append(f"# Konvertiere für Spracherkennung")
                self.result_text.append(f"print('Verarbeite Audio...')")
                self.result_text.append(f"audio_int16 = (audio * 32767).astype(np.int16)")
                self.result_text.append(f"if len(audio_int16.shape) > 1 and audio_int16.shape[1] > 1:")
                self.result_text.append(f"    audio_int16 = np.mean(audio_int16, axis=1).astype(np.int16)  # Mono")
                self.result_text.append(f"")
                self.result_text.append(f"audio_data = sr.AudioData(audio_int16.tobytes(), int(SAMPLERATE), 2)")
                self.result_text.append(f"")
                self.result_text.append(f"# Text erkennen")
                self.result_text.append(f"print('Spracherkennung läuft...')")
                self.result_text.append(f"r = sr.Recognizer()")
                self.result_text.append(f"try:")
                self.result_text.append(f"    text = r.recognize_google(audio_data, language='de-DE')")
                self.result_text.append(f"    print(f'✅ Erkannter Text: {{text}}')")
                self.result_text.append(f"except sr.UnknownValueError:")
                self.result_text.append(f"    print('⚠️ Konnte Sprache nicht verstehen')")
                self.result_text.append(f"except sr.RequestError as e:")
                self.result_text.append(f"    print(f'❌ Fehler bei Spracherkennung: {{e}}')")
                self.result_text.append(f"")
                self.result_text.append(f"# KRITISCHE PARAMETER FÜR ERFOLGREICHE AUFNAHME:")
                self.result_text.append(f"# - Host API: {host_api} (MME funktioniert mit Buds3 Pro!)")
                self.result_text.append(f"# - Latenz: {stream_latency:.6f}s" if stream_latency else "# - Latenz: Default")
                self.result_text.append(f"# - Warmup: 300-500ms (KRITISCH für Bluetooth!)")
                self.result_text.append(f"# - Daten-Timing: NUR vor Countdown leeren, NIE danach!")
                self.result_text.append(f"# - Aufnahmedauer: 10s (genug für vollständige Sätze)")
                self.result_text.append(f"# ")
                self.result_text.append(f"# HÄUFIGE FEHLER:")
                self.result_text.append(f"# 1. recorded_data nach Countdown leeren → Keine Aufnahme!")
                self.result_text.append(f"# 2. Zu kurzer/kein Warmup → Mikrofon nicht bereit")
                self.result_text.append(f"# 3. Audio nach Lautstärke filtern → Leise Sprache verloren")
                self.result_text.append(f"# 4. Stream nicht neu starten → Zweiter Test schlägt fehl")
                self.result_text.append(f"```")
            else:
                logger.warning(f"⚠️ LOW/NO SIGNAL: max_val={max_val:.6f} < threshold=0.01")
                self.result_text.append(f"\n⚠️ WARNUNG: Sehr leises oder kein Signal!")
                self.result_text.append(f"Bitte prüfen Sie die Mikrofoneinstellungen.")
        else:
            logger.error(f"❌ NO DATA RECORDED! recorded_data is empty or None")
            logger.error(f"recorded_data: {self.recorded_data}")
            self.result_text.append(f"\n❌ FEHLER: Keine Daten aufgenommen!")
            self.play_btn.setEnabled(False)  # Disable playback if no data
            self.save_btn.setEnabled(False)  # Disable save if no data
    
    def play_recording(self):
        """Play back the last recorded audio"""
        if self.last_recording is None or len(self.last_recording) == 0:
            self.result_text.append("\n⚠️ Keine Aufnahme zum Abspielen vorhanden!")
            return
        
        try:
            # Disable button during playback
            self.play_btn.setEnabled(False)
            self.play_btn.setText("🔊 Spielt ab...")
            self.result_text.append("\n🔊 Wiedergabe der Aufnahme...")
            
            # Play the recording
            sd.play(self.last_recording, self.last_samplerate)
            
            # Calculate duration for re-enabling button
            duration_ms = int((len(self.last_recording) / self.last_samplerate) * 1000)
            
            # Re-enable button after playback
            QTimer.singleShot(duration_ms + 100, lambda: self.reset_play_button())
            
            logger.info(f"Playing recording: {len(self.last_recording)} samples at {self.last_samplerate} Hz")
            
        except Exception as e:
            logger.error(f"Playback error: {e}")
            self.result_text.append(f"\n❌ Wiedergabe-Fehler: {str(e)[:100]}")
            self.reset_play_button()
    
    def reset_play_button(self):
        """Reset play button after playback"""
        self.play_btn.setEnabled(True)
        self.play_btn.setText("🔊 Anhören")
        self.result_text.append("✅ Wiedergabe beendet")
    
    def save_recording(self):
        """Save the last recording as WAV file"""
        if self.last_recording is None or len(self.last_recording) == 0:
            self.result_text.append("\n⚠️ Keine Aufnahme zum Speichern vorhanden!")
            return
        
        try:
            # Create Audios subdirectory if it doesn't exist
            audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audios")
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)
                logger.info(f"Created directory: {audio_dir}")
            
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            device_short = self.device_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")[:20]
            default_filename = f"aufnahme_{device_short}_{timestamp}.wav"
            default_path = os.path.join(audio_dir, default_filename)
            
            # Open file dialog with default path
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Aufnahme speichern als WAV",
                default_path,
                "WAV Audio (*.wav);;Alle Dateien (*.*)"
            )
            
            if file_path:
                # Import scipy for WAV writing
                try:
                    from scipy.io import wavfile
                except ImportError:
                    # Install scipy if not available
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "--quiet"])
                    from scipy.io import wavfile
                
                # Convert float32 to int16 for WAV
                audio_int16 = np.int16(self.last_recording * 32767)
                
                # Write WAV file
                wavfile.write(file_path, int(self.last_samplerate), audio_int16)
                
                # Get file size
                file_size = os.path.getsize(file_path) / 1024  # KB
                duration = len(self.last_recording) / self.last_samplerate
                
                self.result_text.append(f"\n💾 Aufnahme gespeichert:")
                self.result_text.append(f"   Datei: {os.path.basename(file_path)}")
                self.result_text.append(f"   Größe: {file_size:.1f} KB")
                self.result_text.append(f"   Dauer: {duration:.1f} Sekunden")
                self.result_text.append(f"   Samplerate: {int(self.last_samplerate)} Hz")
                
                logger.info(f"Recording saved to {file_path} ({file_size:.1f} KB)")
                
        except Exception as e:
            logger.error(f"Save error: {e}")
            self.result_text.append(f"\n❌ Speichern fehlgeschlagen: {str(e)[:100]}")
    
    def load_saved_config(self):
        """Load saved configuration for this device"""
        try:
            if self.settings.contains(f"{self.config_key}_samplerate"):
                self.saved_config = {
                    'samplerate': int(self.settings.value(f"{self.config_key}_samplerate", 44100)),
                    'channels': int(self.settings.value(f"{self.config_key}_channels", 1)),
                    'latency': self.settings.value(f"{self.config_key}_latency", None)
                }
                if self.saved_config['latency'] == 'None':
                    self.saved_config['latency'] = None
                elif self.saved_config['latency']:
                    self.saved_config['latency'] = float(self.saved_config['latency'])
                
                logger.info(f"Loaded saved config for {self.device_name}: {self.saved_config}")
        except Exception as e:
            logger.warning(f"Could not load saved config: {e}")
            self.saved_config = None
    
    def save_config(self, samplerate, channels, latency):
        """Save successful configuration for this device"""
        try:
            self.settings.setValue(f"{self.config_key}_samplerate", samplerate)
            self.settings.setValue(f"{self.config_key}_channels", channels)
            self.settings.setValue(f"{self.config_key}_latency", str(latency) if latency else 'None')
            self.settings.sync()
            logger.info(f"Saved config for {self.device_name}: sr={samplerate}, ch={channels}, latency={latency}")
        except Exception as e:
            logger.warning(f"Could not save config: {e}")

class BatchTestDialog(QDialog):
    """Dialog for batch testing all microphones"""
    def __init__(self, devices, parent=None):
        super().__init__(parent)
        self.devices = devices
        self.test_results = []
        self.current_device_index = 0
        self.test_duration = 3  # 3 seconds per device
        self.is_testing = False
        
        self.setWindowTitle("📊 Batch-Test aller Mikrofone")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Info label
        self.info_label = QLabel(f"Teste {len(devices)} Geräte automatisch (je {self.test_duration} Sekunden)")
        layout.addWidget(self.info_label)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMaximum(len(devices))
        layout.addWidget(self.progress)
        
        # Current device label
        self.current_label = QLabel("Bereit zum Start...")
        self.current_label.setWordWrap(True)
        layout.addWidget(self.current_label)
        
        # Results table
        self.results_table = QTableView()
        self.results_model = QStandardItemModel()
        self.results_model.setHorizontalHeaderLabels([
            "Gerät", "Status", "Max. Amplitude", "Durchschn.", "Qualität"
        ])
        self.results_table.setModel(self.results_model)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 Test starten")
        self.start_btn.clicked.connect(self.start_batch_test)
        btn_layout.addWidget(self.start_btn)
        
        self.export_btn = QPushButton("💾 Ergebnisse exportieren")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        btn_layout.addWidget(self.export_btn)
        
        self.close_btn = QPushButton("Schließen")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def start_batch_test(self):
        """Start testing all devices"""
        if self.is_testing:
            return
            
        self.is_testing = True
        self.test_results = []
        self.current_device_index = 0
        self.results_model.setRowCount(0)
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        
        logger.info(f"Starting batch test for {len(self.devices)} devices")
        self.test_next_device()
    
    def test_next_device(self):
        """Test the next device in the list"""
        if self.current_device_index >= len(self.devices):
            self.finish_batch_test()
            return
        
        device = self.devices[self.current_device_index]
        device_name = device.get('Name', 'Unknown')
        device_idx = device.get('Index', -1)
        
        self.current_label.setText(f"🎤 Teste: {device_name}...")
        self.progress.setValue(self.current_device_index)
        
        # Test the device
        QTimer.singleShot(100, lambda: self.test_device(device_idx, device_name))
    
    def test_device(self, device_idx, device_name):
        """Test a single device"""
        logger.info(f"Testing device: {device_name} (idx: {device_idx})")
        
        result = {
            'name': device_name,
            'index': device_idx,
            'status': 'Fehler',
            'max_amplitude': 0.0,
            'mean_amplitude': 0.0,
            'quality': 0,
            'error': None
        }
        
        try:
            # Try to record from the device
            device = sd.query_devices(device_idx)
            samplerate = int(device.get('default_samplerate', 44100))
            channels = min(1, device.get('max_input_channels', 1))
            
            # Try different configurations
            configurations = [
                (samplerate, channels),
                (44100, channels),
                (48000, channels),
                (16000, channels)
            ]
            
            recording = None
            for sr, ch in configurations:
                try:
                    # Quick test recording
                    recording = sd.rec(
                        int(self.test_duration * sr),
                        samplerate=sr,
                        channels=ch,
                        device=device_idx,
                        dtype='float32'
                    )
                    sd.wait()
                    break
                except:
                    continue
            
            if recording is not None and len(recording) > 0:
                # Analyze recording
                max_amp = float(np.abs(recording).max())
                mean_amp = float(np.abs(recording).mean())
                
                result['max_amplitude'] = max_amp
                result['mean_amplitude'] = mean_amp
                
                # Calculate quality score (0-100)
                # Check for silence/noise ratio
                silence_threshold = 0.0001
                
                if max_amp < silence_threshold:
                    # Complete silence - no microphone or not working
                    quality = 0
                    result['status'] = '❌ Kein Signal'
                elif max_amp > 0.5:  # Too loud, might be clipping
                    quality = 50
                    result['status'] = '⚠️ Zu laut'
                elif max_amp > 0.01 and mean_amp > 0.0001:
                    # Good signal with activity
                    # Calculate SNR-like metric
                    snr_factor = max_amp / max(mean_amp, 0.0001)
                    if snr_factor > 10:  # Good peaks relative to average
                        quality = 90
                    elif snr_factor > 5:
                        quality = 70
                    else:
                        quality = 50
                    result['status'] = '✅ Funktioniert'
                elif max_amp > 0.001:
                    # Weak but present signal
                    quality = 30
                    result['status'] = '⚠️ Schwaches Signal'
                else:
                    # Very weak - probably just noise
                    quality = 10
                    result['status'] = '⚠️ Nur Rauschen'
                
                result['quality'] = quality
            else:
                result['status'] = '❌ Fehler'
                result['quality'] = 0
                
        except Exception as e:
            result['error'] = str(e)
            result['status'] = '❌ Fehler'
            logger.error(f"Error testing {device_name}: {e}")
        
        # Add result to table
        self.add_result_to_table(result)
        self.test_results.append(result)
        
        # Continue with next device
        self.current_device_index += 1
        QTimer.singleShot(500, self.test_next_device)  # Small delay between tests
    
    def add_result_to_table(self, result):
        """Add test result to the table"""
        row = [
            QStandardItem(result['name'][:50]),
            QStandardItem(result['status']),
            QStandardItem(f"{result['max_amplitude']:.4f}"),
            QStandardItem(f"{result['mean_amplitude']:.6f}"),
            QStandardItem(f"{result['quality']}%")
        ]
        
        # Color code by quality
        if result['quality'] >= 70:
            color = QColor(0, 200, 0)  # Green
        elif result['quality'] >= 40:
            color = QColor(255, 165, 0)  # Orange
        else:
            color = QColor(255, 0, 0)  # Red
        
        row[4].setBackground(color)
        self.results_model.appendRow(row)
    
    def finish_batch_test(self):
        """Finish the batch test and sort results"""
        self.is_testing = False
        self.progress.setValue(len(self.devices))
        self.current_label.setText("✅ Test abgeschlossen!")
        self.start_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        # Sort results by quality
        self.test_results.sort(key=lambda x: x['quality'], reverse=True)
        
        # Rebuild table sorted
        self.results_model.setRowCount(0)
        for result in self.test_results:
            self.add_result_to_table(result)
        
        # Show summary and offer to filter
        working = sum(1 for r in self.test_results if r['quality'] > 30)
        non_working = sum(1 for r in self.test_results if r['quality'] == 0)
        total = len(self.test_results)
        
        self.info_label.setText(
            f"Test abgeschlossen: {working}/{total} funktionieren | {non_working} ohne Signal (Qualität 0%)"
        )
        
        # Ask if user wants to hide non-working devices
        if non_working > 0:
            reply = QMessageBox.question(
                self, 
                "Nicht-funktionierende Geräte gefunden",
                f"{non_working} Geräte haben kein Signal (0% Qualität).\n\n"
                f"Möchten Sie diese aus der Hauptliste ausblenden?\n\n"
                f"(Sie können sie später durch 'Alle' im Filter wieder anzeigen)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Store non-working devices for filtering
                self.parent().non_working_devices = [r['name'] for r in self.test_results if r['quality'] == 0]
                self.parent().apply_device_filter()
                QMessageBox.information(self, "Filter angewendet", 
                    f"{non_working} nicht-funktionierende Geräte wurden ausgeblendet.\n\n"
                    f"Wählen Sie 'Alle' im Filter-Dropdown um sie wieder anzuzeigen.")
        
        logger.info(f"Batch test complete: {working}/{total} devices working, {non_working} non-functional")
    
    def export_results(self):
        """Export test results to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_test_results_{timestamp}.csv"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Testergebnisse exportieren",
                filename,
                "CSV Dateien (*.csv);;Alle Dateien (*.*)"
            )
            
            if file_path:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Gerät', 'Status', 'Max Amplitude', 'Durchschnitt', 'Qualität %'])
                    for result in self.test_results:
                        writer.writerow([
                            result['name'],
                            result['status'],
                            f"{result['max_amplitude']:.4f}",
                            f"{result['mean_amplitude']:.6f}",
                            result['quality']
                        ])
                
                QMessageBox.information(self, "Erfolg", f"Ergebnisse exportiert nach:\n{os.path.basename(file_path)}")
                
        except Exception as e:
            QMessageBox.warning(self, "Fehler", f"Export fehlgeschlagen: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 Aktive Mikrofone und verfügbare Eingabegeräte")
        self.resize(1200, 800)
        self.settings = QSettings('MikrofoneTool', 'Settings')
        self.dark_mode = self.settings.value('dark_mode', False, type=bool)
        self.auto_refresh = False
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh)
        self.all_devices = []
        
        # Hot-plug detection
        self.device_monitor_timer = QTimer()
        self.device_monitor_timer.timeout.connect(self.check_device_changes)
        self.device_monitor_timer.start(2000)  # Check every 2 seconds
        self.last_device_count = 0
        self.last_device_names = set()
        
        # List of non-working devices from batch test
        self.non_working_devices = []
        self.model = QStandardItemModel()
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.view.setSortingEnabled(True)
        self.view.horizontalHeader().setStretchLastSection(False)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_context_menu)
        self.status = QLabel()
        self.status.setText("Bereit")
        
        # Suchfeld für Filterung
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("🔍 Suche nach Gerätename, Host API...")
        self.search_field.textChanged.connect(self.filter_devices)
        self.search_field.setClearButtonEnabled(True)
        
        # Filteroptionen
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Alle Geräte", "Nur funktionierende", "Nur nutzbare", "Nur Standard", "Nur Bluetooth", "Nur USB"])
        self.filter_combo.currentTextChanged.connect(self.filter_devices)
        
        # List of non-working devices from batch test
        self.non_working_devices = []
        
        # Auto-Refresh Checkbox
        self.auto_refresh_check = QCheckBox("Auto-Refresh (5s)")
        self.auto_refresh_check.toggled.connect(self.toggle_auto_refresh)
        
        # Dark Mode Toggle
        self.dark_mode_check = QCheckBox("🌙 Dark Mode")
        self.dark_mode_check.setChecked(self.dark_mode)
        self.dark_mode_check.toggled.connect(self.toggle_dark_mode)
        
        self.btnRefresh = QPushButton("🔄 Aktualisieren (F5)")
        self.btnBatchTest = QPushButton("🚀 Batch-Test")
        self.btnBatchTest.setToolTip("Teste alle Mikrofone automatisch und sortiere nach Qualität")
        self.btnExport = QPushButton("📄 Als HTML exportieren")
        self.btnExportCSV = QPushButton("📊 Als CSV exportieren")
        self.btnExportJSON = QPushButton("📋 Als JSON exportieren")
        self.btnClose = QPushButton("❌ Schließen")
        self.btnRefresh.clicked.connect(self.refresh)
        self.btnBatchTest.clicked.connect(self.start_batch_test)
        self.btnExport.clicked.connect(self.export_html)
        self.btnExportCSV.clicked.connect(self.export_csv)
        self.btnExportJSON.clicked.connect(self.export_json)
        self.btnClose.clicked.connect(self.close)
        central = QWidget()
        layout = QVBoxLayout()
        
        # Header mit Info
        header = QLabel("📋 Liste der aktuell aktiven und nutzbaren Mikrofone. Doppelklick auf Spaltenköpfe sortiert. Rechtsklick für Kontextmenü.")
        header.setWordWrap(True)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(header)
        
        # Filter-Leiste
        filter_bar = QHBoxLayout()
        filter_bar.addWidget(QLabel("Filter:"))
        filter_bar.addWidget(self.search_field, 2)
        filter_bar.addWidget(self.filter_combo)
        filter_bar.addWidget(self.auto_refresh_check)
        filter_bar.addWidget(self.dark_mode_check)
        filter_bar.addStretch()
        layout.addLayout(filter_bar)
        layout.addWidget(self.view)
        bar = QHBoxLayout()
        bar.addWidget(self.status)
        bar.addStretch(1)
        bar.addWidget(self.btnRefresh)
        bar.addWidget(self.btnBatchTest)
        bar.addWidget(self.btnExport)
        bar.addWidget(self.btnExportCSV)
        bar.addWidget(self.btnExportJSON)
        bar.addWidget(self.btnClose)
        layout.addLayout(bar)
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.collector = None
        self.columns = ["Index","Name","Host API","Max Input Channels","Default Samplerate","Low Input Latency","High Input Latency","Ist Standard-Eingabe","Nutzbar (Test)","Rohdaten"]
        self.build_empty_model()
        self.apply_theme()
        self.refresh()

        export_action = QAction("Exportieren", self)
        export_action.triggered.connect(self.export_html)
        self.addAction(export_action)
        export_action.setShortcut("Ctrl+E")
        refresh_action = QAction("Aktualisieren", self)
        refresh_action.triggered.connect(self.refresh)
        self.addAction(refresh_action)
        refresh_action.setShortcut("F5")

    def show_context_menu(self, position):
        if not self.view.selectionModel().hasSelection():
            return
        
        row = self.view.selectionModel().currentIndex().row()
        if row < 0:
            return
            
        device_idx = int(self.model.item(row, 0).text())
        device_name = self.model.item(row, 1).text()
        is_bluetooth = "Bluetooth" in self.model.item(row, 2).text()
        
        menu = QMenu(self)
        
        set_default_action = QAction(f"Als Standard-Mikrofon setzen: {device_name}", self)
        set_default_action.triggered.connect(lambda: self.set_default_device(device_idx, device_name))
        menu.addAction(set_default_action)
        
        test_action = QAction(f"Mikrofon testen: {device_name}", self)
        test_action.triggered.connect(lambda: self.test_microphone(device_idx, device_name))
        menu.addAction(test_action)
        
        if is_bluetooth:
            menu.addSeparator()
            bluetooth_info = QAction("Bluetooth-Gerät erkannt", self)
            bluetooth_info.setEnabled(False)
            menu.addAction(bluetooth_info)
        
        menu.exec(self.view.viewport().mapToGlobal(position))
    
    def set_default_device(self, device_idx, device_name):
        try:
            sd.default.device = (device_idx, sd.default.device[1])
            QMessageBox.information(self, "Erfolg", f"Standard-Mikrofon geändert zu:\n{device_name}\n\nBitte Aktualisieren drücken.")
            self.refresh()
        except Exception as e:
            QMessageBox.warning(self, "Fehler", f"Konnte Standard-Mikrofon nicht ändern:\n{str(e)}")
    
    def test_microphone(self, device_idx, device_name):
        # Direkt testen ohne unnötige Vorschläge
        dialog = MicrophoneTestDialog(device_idx, device_name, self)
        dialog.exec()
    
    def start_batch_test(self):
        """Start batch testing all microphones"""
        if not self.all_devices:
            QMessageBox.information(self, "Info", "Keine Geräte zum Testen gefunden.\nBitte zuerst Aktualisieren drücken.")
            return
        
        # Filter only input devices
        input_devices = [d for d in self.all_devices if d.get('Max Input Channels', 0) > 0]
        
        if not input_devices:
            QMessageBox.information(self, "Info", "Keine Eingabegeräte gefunden.")
            return
        
        # Start batch test dialog
        dialog = BatchTestDialog(input_devices, self)
        dialog.exec()
    
    def check_device_changes(self):
        """Check for device additions or removals (hot-plug detection)"""
        try:
            # Get current devices quietly
            devices = sd.query_devices()
            current_device_count = len([d for d in devices if d.get('max_input_channels', 0) > 0])
            current_device_names = set([d.get('name', '') for d in devices if d.get('max_input_channels', 0) > 0])
            
            # Check if devices changed
            if (current_device_count != self.last_device_count or 
                current_device_names != self.last_device_names):
                
                # Determine what changed
                added = current_device_names - self.last_device_names
                removed = self.last_device_names - current_device_names
                
                # Update stored values
                self.last_device_count = current_device_count
                self.last_device_names = current_device_names
                
                # Show notification and refresh
                if added or removed:
                    status_msg = []
                    if added:
                        for device in added:
                            status_msg.append(f"🔌 Neues Gerät: {device[:30]}")
                            logger.info(f"Device added: {device}")
                    if removed:
                        for device in removed:
                            status_msg.append(f"🔍 Gerät entfernt: {device[:30]}")
                            logger.info(f"Device removed: {device}")
                    
                    self.status.setText(" | ".join(status_msg[:2]))  # Show max 2 messages
                    
                    # Auto-refresh the device list
                    self.refresh()
                    
                    # Clear status after 5 seconds
                    QTimer.singleShot(5000, lambda: self.status.setText("Bereit"))
                    
        except Exception as e:
            # Silently ignore errors in monitoring
            pass

    def build_empty_model(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(self.columns)

    def refresh(self):
        if self.collector and self.collector.isRunning():
            self.status.setText("⚠️ Aktualisierung läuft bereits...")
            return
        self.status.setText("🔍 Suche Geräte...")
        self.btnRefresh.setEnabled(False)
        self.build_empty_model()
        self.collector = DeviceCollector()
        self.collector.resultReady.connect(self.on_result)
        self.collector.error.connect(self.on_error)
        self.collector.finished.connect(lambda: self.btnRefresh.setEnabled(True))
        self.collector.start()

    def on_error(self, msg):
        self.status.setText(f"❌ {msg}")
        self.btnRefresh.setEnabled(True)
        QMessageBox.critical(self, "Fehler bei Geräteerkennung", 
                           f"Es ist ein Fehler aufgetreten:\n\n{msg}\n\n"
                           "Mögliche Lösungen:\n"
                           "• Prüfen Sie die Audiogeräte-Einstellungen\n"
                           "• Starten Sie die Anwendung neu\n"
                           "• Schließen Sie andere Audio-Anwendungen")

    def toggle_auto_refresh(self, checked):
        self.auto_refresh = checked
        if checked:
            self.refresh_timer.start(5000)  # 5 Sekunden
            self.status.setText("Auto-Refresh aktiviert (alle 5 Sekunden)")
        else:
            self.refresh_timer.stop()
            self.status.setText("Auto-Refresh deaktiviert")
    
    def toggle_dark_mode(self, checked):
        self.dark_mode = checked
        self.settings.setValue('dark_mode', checked)
        self.apply_theme()
    
    def apply_theme(self):
        if self.dark_mode:
            # Dark Mode Style
            self.setStyleSheet("""
                QMainWindow { background-color: #2b2b2b; color: #ffffff; }
                QTableView { background-color: #1e1e1e; color: #ffffff; gridline-color: #555; 
                            alternate-background-color: #2a2a2a; selection-background-color: #4a4a4a; }
                QHeaderView::section { background-color: #3a3a3a; color: #ffffff; 
                                      border: 1px solid #555; padding: 4px; }
                QLabel { color: #ffffff; }
                QPushButton { background-color: #4a4a4a; color: #ffffff; border: 1px solid #666;
                            padding: 5px 10px; border-radius: 3px; }
                QPushButton:hover { background-color: #5a5a5a; }
                QPushButton:pressed { background-color: #3a3a3a; }
                QLineEdit { background-color: #3a3a3a; color: #ffffff; border: 1px solid #555;
                          padding: 5px; border-radius: 3px; }
                QComboBox { background-color: #3a3a3a; color: #ffffff; border: 1px solid #555;
                          padding: 5px; border-radius: 3px; }
                QComboBox::drop-down { border: none; }
                QComboBox::down-arrow { image: none; border-left: 5px solid transparent;
                                       border-right: 5px solid transparent; border-top: 5px solid #fff; }
                QCheckBox { color: #ffffff; }
                QMenu { background-color: #2b2b2b; color: #ffffff; }
                QMenu::item:selected { background-color: #4a4a4a; }
                QProgressBar { background-color: #3a3a3a; border: 1px solid #555; border-radius: 3px; }
                QProgressBar::chunk { background-color: #5a9fd4; }
                QTextEdit { background-color: #1e1e1e; color: #ffffff; border: 1px solid #555; }
            """)
        else:
            # Light Mode (Standard)
            self.setStyleSheet("")
    
    def apply_device_filter(self):
        """Apply filter to hide non-working devices"""
        if self.non_working_devices:
            self.filter_combo.setCurrentText("Nur funktionierende")
            self.filter_devices()
    
    def filter_devices(self):
        search_text = self.search_field.text().lower()
        filter_type = self.filter_combo.currentText()
        
        for row in range(self.model.rowCount()):
            should_show = True
            
            # Text-Suche
            if search_text:
                row_text = ""
                for col in range(self.model.columnCount()):
                    item = self.model.item(row, col)
                    if item:
                        row_text += item.text().lower() + " "
                should_show = search_text in row_text
            
            # Filter-Typ
            if should_show and filter_type != "Alle Geräte":
                if filter_type == "Nur funktionierende":
                    # Hide devices identified as non-working in batch test
                    name_item = self.model.item(row, self.columns.index("Name"))
                    if name_item and self.non_working_devices:
                        device_name = name_item.text()
                        should_show = device_name not in self.non_working_devices
                elif filter_type == "Nur nutzbare":
                    item = self.model.item(row, self.columns.index("Nutzbar (Test)"))
                    should_show = item and item.text() == "Ja"
                elif filter_type == "Nur Standard":
                    item = self.model.item(row, self.columns.index("Ist Standard-Eingabe"))
                    should_show = item and item.text() == "Ja"
                elif filter_type == "Nur Bluetooth":
                    item = self.model.item(row, self.columns.index("Host API"))
                    should_show = item and "bluetooth" in item.text().lower()
                elif filter_type == "Nur USB":
                    item = self.model.item(row, self.columns.index("Name"))
                    should_show = item and ("usb" in item.text().lower() or "USB" in item.text())
            
            self.view.setRowHidden(row, not should_show)
        
        # Update Status
        visible_count = sum(1 for r in range(self.model.rowCount()) if not self.view.isRowHidden(r))
        total = self.model.rowCount()
        self.status.setText(f"Zeige {visible_count} von {total} Geräten")
    
    def on_result(self, devices, _):
        self.all_devices = devices
        
        # Update device tracking for hot-plug detection
        if not hasattr(self, 'last_device_count'):
            self.last_device_count = len(devices)
            self.last_device_names = set([d.get('Name', '') for d in devices])
        for rec in devices:
            row_items = []
            for col in self.columns:
                val = rec.get(col, "")
                if col == "Rohdaten":
                    # Index aus Rohdaten entfernen, da er bereits eine eigene Spalte hat
                    raw_data = rec.get(col, {}).copy() if isinstance(rec.get(col, {}), dict) else {}
                    raw_data.pop('index', None)
                    val = json.dumps(raw_data, ensure_ascii=False)
                if isinstance(val, bool):
                    text = "Ja" if val else "Nein"
                else:
                    text = "" if val is None else str(val)
                item = QStandardItem(text)
                if col in ("Index","Max Input Channels"):
                    item.setData(val if isinstance(val, int) else None, Qt.ItemDataRole.UserRole)
                row_items.append(item)
            self.model.appendRow(row_items)
            if rec.get("Ist Standard-Eingabe", False):
                for it in row_items:
                    it.setBackground(QColor(230, 255, 230))
            elif not rec.get("Nutzbar (Test)", True):
                for it in row_items:
                    it.setBackground(QColor(255, 235, 235))
        self.view.resizeColumnsToContents()
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        total = self.model.rowCount()
        nutzbar = sum(1 for r in range(total) if self.model.item(r, self.columns.index("Nutzbar (Test)")).text() == "Ja")
        default_count = sum(1 for r in range(total) if self.model.item(r, self.columns.index("Ist Standard-Eingabe")).text() == "Ja")
        self.status.setText(f"📊 Gefundene Mikrofone: {total} | ✅ Nutzbar: {nutzbar} | ⭐ Standard: {default_count}")
        
        # Filter anwenden falls aktiv
        self.filter_devices()

    def export_html(self):
        rows = []
        for r in range(self.model.rowCount()):
            row = []
            for c in range(self.model.columnCount()):
                row.append(self.model.item(r, c).text())
            rows.append(row)
        html = []
        html.append("<!DOCTYPE html><html lang='de'><head><meta charset='utf-8'><title>Mikrofonliste</title>")
        html.append("<style>body{font-family:Segoe UI,Arial,sans-serif;margin:20px;background:#f7f7f7}h1{font-size:20px}table{border-collapse:collapse;width:100%;background:#fff}th,td{border:1px solid #ddd;padding:8px;vertical-align:top}th{background:#fafafa;position:sticky;top:0}tr:nth-child(even){background:#f9f9f9}.ok{background:#eaffea}.bad{background:#ffefef}.meta{color:#555;font-size:12px}</style>")
        html.append("</head><body>")
        html.append("<h1>Aktive und nutzbare Mikrofone</h1>")
        html.append("<div class='meta'>Erzeugt mit Python am Standort des Skripts.</div>")
        html.append("<table><thead><tr>")
        for col in self.columns:
            html.append(f"<th>{col}</th>")
        html.append("</tr></thead><tbody>")
        idx_def = self.columns.index("Ist Standard-Eingabe")
        idx_use = self.columns.index("Nutzbar (Test)")
        for r, row in enumerate(rows, start=1):
            cls = ""
            if row[idx_def] == "Ja":
                cls = " class='ok'"
            elif row[idx_use] == "Nein":
                cls = " class='bad'"
            html.append(f"<tr{cls}>")
            for cell in row:
                if len(cell) > 2000:
                    cell = cell[:2000] + " …"
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody></table></body></html>")
        html_str = "".join(html)
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = os.path.join(script_dir, "mikrofone_report.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_str)
        try:
            os.startfile(path)
        except Exception:
            pass
        self.status.setText(f"✅ HTML exportiert: {path}")
    
    def export_csv(self):
        """Export als CSV-Datei"""
        import csv
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = os.path.join(script_dir, "mikrofone_report.csv")
        
        with open(path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            
            # Header
            writer.writerow(self.columns[:-1])  # Ohne Rohdaten
            
            # Daten
            for r in range(self.model.rowCount()):
                if not self.view.isRowHidden(r):  # Nur sichtbare Zeilen
                    row = []
                    for c in range(self.model.columnCount() - 1):  # Ohne Rohdaten
                        item = self.model.item(r, c)
                        row.append(item.text() if item else "")
                    writer.writerow(row)
        
        try:
            os.startfile(path)
        except Exception:
            pass
        self.status.setText(f"✅ CSV exportiert: {path}")
    
    def export_json(self):
        """Export als JSON-Datei mit allen Details"""
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = os.path.join(script_dir, "mikrofone_report.json")
        
        export_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_devices": len(self.all_devices),
            "devices": self.all_devices
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        try:
            os.startfile(path)
        except Exception:
            pass
        self.status.setText(f"✅ JSON exportiert: {path}")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("MIKROPHON TEST APPLICATION STARTED")
    logger.info(f"Log file: {log_filename}")
    logger.info("="*60)
    
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())