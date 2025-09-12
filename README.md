# Diktieren - Speech-to-Text Dictation App

Eine Windows-Desktop-Anwendung für Spracheingabe mit umfassender Bluetooth-Unterstützung und automatischer Geräteerkennung.

## Features

- **Spracherkennung** in mehreren Sprachen (Deutsch, Englisch, Russisch, Französisch, Spanisch)
- **Echtzeit-Waveform-Visualisierung** - Live-Anzeige der Audio-Wellenform während der Aufnahme
- **Automatische Geräte-Erkennung** mit Qualitätsmessung (0-100%)
- **Bluetooth-Unterstützung** mit spezieller Optimierung für Samsung Galaxy Buds und andere Bluetooth-Kopfhörer
- **Hot-Plug-Erkennung** - Geräte werden automatisch erkannt beim Anschließen/Trennen
- **Intelligente Geräte-Filterung** - Nicht-funktionierende Geräte werden automatisch ausgefiltert
- **Konfigurations-Speicherung** - Einstellungen werden pro Gerät gespeichert
- **Erweiterte Test-Funktionen** für Mikrofon-Debugging
- **Audio-Level-Monitor** - Visuelle Pegelanzeige in Echtzeit

## Installation

### Voraussetzungen

- Windows 10/11
- Python 3.8 oder höher
- Mikrofon (USB, Bluetooth oder eingebaut)

### Setup

1. Repository klonen:
```bash
git clone https://github.com/Moddy14/Diktieren.git
cd Diktieren
```

2. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

3. Anwendung starten:
```bash
python Diktieren.py
```

## Verwendung

1. **Gerät auswählen**: Die App zeigt alle verfügbaren Mikrofone mit Qualitätsprozenten an
2. **Sprache wählen**: Deutsch, Englisch oder Auto-Erkennung
3. **Start klicken**: Nach 3-Sekunden-Countdown beginnt die Aufnahme
4. **Sprechen**: Der erkannte Text erscheint im Textfeld
5. **Stop klicken**: Beendet die Aufnahme

### Tastenkürzel

- `Strg+S`: Aufnahme starten/stoppen
- `Strg+C`: Text kopieren
- `Strg+A`: Alles auswählen

## Bluetooth-Geräte

Die App wurde speziell für Bluetooth-Kopfhörer optimiert:

- Automatische Warmup-Phase für Bluetooth-Geräte
- Spezielle Unterstützung für Samsung Galaxy Buds
- WDM-KS Fehler-Erkennung und -Filterung
- Kontinuierliche Aufnahme statt Threshold-basiert für bessere Bluetooth-Kompatibilität

## Troubleshooting

### Gerät wird nicht erkannt
- "Refresh" klicken um Geräteliste zu aktualisieren
- Bluetooth-Gerät neu verbinden
- Windows-Audioeinstellungen prüfen

### Keine Spracherkennung
- Lautstärke prüfen (Qualitätsprozente sollten > 0% sein)
- Andere Sprache probieren
- Test-Dialog öffnen für erweiterte Diagnose

### WDM-KS Fehler
- Geräte mit WDM-KS Fehlern werden automatisch ausgefiltert
- Alternative Geräte werden automatisch vorgeschlagen

## Konfiguration

Die App speichert Einstellungen in:
- `threshold_config.json` - Schwellenwerte pro Gerät
- `device_configs.json` - Arbeitsende Konfigurationen

## Entwicklung

### Projekt-Struktur
```
Diktieren/
├── Diktieren.py          # Hauptanwendung
├── requirements.txt      # Python-Abhängigkeiten
├── threshold_config.json # Gespeicherte Schwellenwerte
├── device_configs.json   # Gespeicherte Gerätekonfigurationen
├── CLAUDE.md            # Entwicklungs-Dokumentation
└── SprachEingabe.log    # Debug-Log
```

### Bekannte Limitierungen

- Windows-only (PyQt6 und sounddevice Windows-APIs)
- Offline-Spracherkennung (keine Cloud-Services)
- Maximale Aufnahmequalität abhängig vom Mikrofon

## Lizenz

MIT License - siehe [LICENSE](LICENSE) Datei

## Beiträge

Contributions sind willkommen! Bitte erstelle einen Pull Request oder öffne ein Issue.

## Autor

Heinrich Moddy

## Acknowledgments

- Google Speech Recognition API für die Spracherkennung
- PyQt6 für die GUI
- sounddevice für Audio-Verarbeitung