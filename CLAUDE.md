# Diktieren - Speech Recognition App - Entwickler-Dokumentation

## Überblick
Eine robuste Windows-Desktop-Anwendung für Spracheingabe mit umfassender Bluetooth-Unterstützung, automatischer Geräteerkennung und intelligenter Qualitätsmessung.

## Architektur

### Hauptkomponenten
- **Diktieren.py**: Hauptanwendung mit PyQt6 GUI
- **sounddevice**: Primäre Audio-Bibliothek für Stream-Management
- **speech_recognition**: Google Speech Recognition für Spracherkennung
- **Hybrid-Ansatz**: Kombiniert sounddevice und speech_recognition für maximale Kompatibilität

### Geräte-Management

#### Automatische Geräte-Erkennung
1. **Startup-Scan**: Testet alle Geräte beim Start
2. **Qualitätsmessung**: 0-100% basierend auf Signal-Stärke (logarithmische Skala)
3. **Stream-Test**: Verifiziert dass Geräte tatsächlich funktionieren
4. **WDM-KS Filterung**: Erkennt und filtert problematische Windows-Audio-Treiber

#### Hot-Plug Detection
- Alle 3 Sekunden Überprüfung auf neue/entfernte Geräte
- Automatisches Refresh ohne Neustart
- Verhindert doppeltes Scannen beim Start

### Audio-Pipeline

#### Aufnahme-Workflow
1. **Hardware-Warmup** (300ms) - Aktiviert Bluetooth-Geräte
2. **Stream-Warmup** (500ms) - Stabilisiert Audio-Stream
3. **Countdown** (3 Sekunden) - Mit kontinuierlichem Stream-Read
4. **Kontinuierliche Aufnahme** - 3-Sekunden-Segmente
5. **Spracherkennung** - Google Speech API (offline)

#### Bluetooth-Spezialbehandlung
```python
# Bluetooth benötigt spezielle Behandlung:
- Warmup-Reads zum Aktivieren
- Kontinuierliche Aufnahme statt Threshold-basiert
- Niedrige Thresholds (1-50)
- Dynamic Threshold MUSS deaktiviert sein
```

### Konfigurations-System

#### threshold_config.json
```json
{
  "device_index": threshold_value,
  // Beispiel:
  "1": 50,   // MME Bluetooth
  "9": 50,   // DirectSound Bluetooth
  "21": 200  // WASAPI höhere Threshold
}
```

#### device_configs.json
```json
{
  "device_name": {
    "samplerate": 44100,
    "channels": 1,
    "latency": 0.03483,
    "host_api": "MME"
  }
}
```

## Kritische Implementierungsdetails

### Geräte-Filterung beim Start
```python
# Teste Audio-Signal (0.3s Recording)
max_val = np.max(np.abs(recording))
if max_val < 0.0001:  # Kein Signal
    return False

# Qualität berechnen (logarithmisch)
db = 20 * np.log10(max_val)
quality = max(0, min(100, int((db + 40) * 2.5)))

# Stream-Test für WDM-KS Probleme
try:
    test_stream = sd.InputStream(...)
    test_stream.close()
except:
    # WDM-KS Fehler erkennen und Gerät ausfiltern
```

### Fallback-Strategien
1. **Primär**: sounddevice mit gespeicherter Konfiguration
2. **Sekundär**: speech_recognition Microphone
3. **Auto-Switch**: Bei Fehler automatisch nächstes Gerät

### Test-Dialog Features
- **Einzeltest**: Detaillierte Analyse eines Geräts
- **Batch-Test**: Alle Geräte mit Qualitätsmessung
- **Live-Monitor**: Echtzeit Audio-Level Anzeige
- **Threshold-Anpassung**: Manuell oder automatisch

## Bekannte Probleme & Lösungen

### Problem: Near-Zero Audio bei Bluetooth
**Symptom**: Audio-Level 0.0000-0.0001
**Lösung**: Warmup-Phase mit mehreren kurzen Reads

### Problem: WDM-KS "Invalid device" Fehler
**Symptom**: Geräte erscheinen verfügbar, schlagen aber beim Stream-Öffnen fehl
**Lösung**: Stream-Test beim Start, automatische Filterung

### Problem: Exponentieller Threshold-Decay
**Symptom**: Threshold steigt kontinuierlich an
**Lösung**: Dynamic Threshold deaktivieren

### Problem: Bluetooth Audio-Verzögerung
**Symptom**: Erste Worte werden abgeschnitten
**Lösung**: 3-Sekunden-Countdown mit kontinuierlichem Stream-Read

## Performance-Optimierungen

1. **Parallel-Scanning**: Geräte werden parallel getestet
2. **Cache-System**: Funktionierende Konfigurationen werden gespeichert
3. **Lazy Loading**: speech_recognition nur als Fallback
4. **Sortierung nach Qualität**: Beste Geräte zuerst in der Liste

## Debug & Entwicklung

### Log-Level
```python
logging.basicConfig(
    level=logging.DEBUG,  # oder INFO für weniger Output
    format='%(asctime)s [%(levelname)s] %(message)s'
)
```

### Test-Skripte
- `test_device9.py`: Direkter Test eines spezifischen Geräts
- `test_comparison.py`: Vergleich PyAudio vs speech_recognition
- `list_all_devices.py`: Zeigt alle verfügbaren Geräte
- `test_warmup.py`: Test der Warmup-Sequenz
- `Mikrophon.Test.py`: Umfassender Geräte-Test

### Wichtige Metriken
- **Signal-Qualität**: 0-100% (> 50% ist gut)
- **Max Amplitude**: > 0.01 für gute Erkennung
- **Latenz**: < 100ms für Echtzeit-Gefühl

## Deployment

### Requirements
```
PyQt6>=6.5.0
sounddevice>=0.4.6
numpy>=1.24.0
SpeechRecognition>=3.10.0
pyaudio>=0.2.11
```

### Windows-spezifisch
- Benötigt Windows Audio Session API (WASAPI)
- DirectSound für bessere Bluetooth-Kompatibilität
- MME als universeller Fallback

## Zukünftige Verbesserungen

1. **Cloud-Speech-APIs**: Optional für bessere Erkennung
2. **Multi-Language Auto-Detect**: Automatische Spracherkennung
3. **Hotkey-Support**: Globale Tastenkürzel
4. **Audio-Preprocessing**: Rauschunterdrückung, Normalisierung
5. **Export-Funktionen**: Text als verschiedene Formate speichern