# Diktieren - Speech Recognition App

## Wichtige Hinweise für Bluetooth-Geräte (Samsung Galaxy Buds3 Pro)

### Problem mit Device 1
Device 1 (MME API) zeigt oft sehr niedrige Audio-Level (0.0000-0.0128) und funktioniert möglicherweise nicht richtig mit Bluetooth-Geräten.

### Lösung: Device 9 verwenden
Falls Device 1 nicht funktioniert:
1. Wähle **Device 9** aus der Geräteliste (DirectSound API)
2. Device 9 nutzt Windows DirectSound statt MME und funktioniert oft besser mit Bluetooth

### Geräte-Übersicht
- **Device 1**: Kopfhörer (Buds3 Pro) - MME API @ 44100Hz
- **Device 9**: Kopfhörer (Buds3 Pro) - DirectSound API @ 44100Hz ✅ (empfohlen)
- **Device 21**: Kopfhörer (Buds3 Pro) - WASAPI @ 16000Hz (funktioniert, aber niedrigere Qualität)

### Threshold-Konfiguration
Die Datei `threshold_config.json` enthält optimierte Schwellenwerte:
- Device 1: 50
- Device 9: 50 
- Device 21: 200

### Wichtige Implementierungsdetails

#### Bluetooth-Warmup
Bluetooth-Geräte benötigen eine Warmup-Phase:
1. Stream öffnen
2. 10 kurze Aufnahmen (0.1s) zum Aktivieren
3. 3-Sekunden-Countdown mit kontinuierlichem Lesen
4. Dann erst normale Aufnahme starten

#### Kontinuierliche Aufnahme
Für Device 1 und 9 wird `recognizer.record()` mit 3-Sekunden-Segmenten verwendet statt `recognizer.listen()`, um Threshold-Probleme zu umgehen.

#### Debug-Informationen
Das Log zeigt:
- max_amp: Maximale Amplitude (Referenz: 0.1035)
- avg_amp: Durchschnittliche Amplitude
- Audio-Level sollte > 0.01 sein für gute Erkennung

### Test-Befehle
```bash
# Teste Device 9 direkt
python test_device9.py

# Vergleiche PyAudio vs speech_recognition
python test_comparison.py

# Liste alle Geräte
python list_all_devices.py
```

### Bekannte Probleme
1. **Near-zero Audio**: Device muss mit Warmup-Reads aktiviert werden
2. **Threshold zu hoch**: Verwende niedrige Werte (1-50) für Bluetooth
3. **Dynamic Threshold**: MUSS FALSE sein, sonst exponentieller Decay!

### Lint und Typecheck Commands
```bash
# Falls verfügbar, führe diese aus:
# npm run lint
# npm run typecheck
# oder
# python -m pylint Diktieren.py
# python -m mypy Diktieren.py
```