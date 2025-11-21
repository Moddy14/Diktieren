# 🎙️ Diktieren - Speech-to-Text Pro

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)

Eine professionelle Windows-Desktop-Anwendung für Spracheingabe mit Fokus auf Bluetooth-Headsets, automatischer Geräteerkennung und Echtzeit-Visualisierung.

## ✨ Features

*   **Multi-Language Support**: Deutsch, Englisch, Russisch, Französisch, Spanisch.
*   **Echtzeit-Visualisierung**: Live-Waveform und Audio-Level-Monitor.
*   **Smart Device Management**:
    *   Automatische Erkennung neuer Geräte (Hot-Plug).
    *   Qualitätsmessung (0-100%) für jedes Mikrofon.
    *   Intelligente Filterung defekter Treiber (WDM-KS).
*   **Bluetooth-Optimierung**: Spezielle Algorithmen für Samsung Galaxy Buds und andere Bluetooth-Headsets (Warmup-Phasen, Latenz-Kompensation).
*   **Persistente Konfiguration**: Speichert Einstellungen pro Gerät.

## 🚀 Quick Start

### Voraussetzungen

*   Windows 10 oder 11
*   Python 3.8+

### Installation

1.  **Repository klonen**
    ```bash
    git clone https://github.com/Moddy14/Diktieren.git
    cd Diktieren
    ```

2.  **Abhängigkeiten installieren**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Starten**
    ```bash
    python Diktieren.py
    ```

## 📖 Dokumentation

*   [Entwickler-Dokumentation](docs/DEVELOPMENT.md) - Architektur und technische Details.
*   [Contributing](CONTRIBUTING.md) - Wie du mithelfen kannst.
*   [Code of Conduct](CODE_OF_CONDUCT.md) - Unsere Verhaltensregeln.

## 🛠️ Verwendung

1.  **Gerät wählen**: Wähle dein Mikrofon aus der Liste (Qualität wird angezeigt).
2.  **Sprache wählen**: Wähle die Zielsprache oder "Auto".
3.  **Start**: Klicke auf "Start" oder drücke `Strg+S`.
4.  **Diktieren**: Sprich nach dem Countdown.
5.  **Stop**: Klicke "Stop" oder drücke erneut `Strg+S`.

## 🤝 Contributing

Beiträge sind willkommen! Bitte lies unsere [Contributing Guidelines](CONTRIBUTING.md) für Details.

## 📄 Lizenz

Dieses Projekt ist unter der MIT Lizenz lizenziert - siehe [LICENSE](LICENSE) für Details.

## 👤 Autor

**Heinrich Moddy**

---
*Built with Python, PyQt6, and Google Speech Recognition.*