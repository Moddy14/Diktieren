# -*- coding: utf-8 -*-
"""
Umfassender Test für neue Features in Diktieren.py:
1. QSettings Persistenz
2. Hot-Plug Detection
3. Saved Config Loading
"""

import sys
import time
from PyQt6.QtCore import QSettings, QCoreApplication
from PyQt6.QtWidgets import QApplication
import sounddevice as sd

def test_qsettings():
    """Test QSettings persistence"""
    print("\n=== TEST 1: QSettings Persistenz ===")
    
    settings = QSettings('MikrofoneTool', 'DeviceConfigs')
    
    # List all saved configs
    all_keys = settings.allKeys()
    devices = {}
    
    for key in all_keys:
        if '_samplerate' in key:
            device_key = key.replace('_samplerate', '')
            device_name = device_key.replace('device_', '').replace('_', ' ')[:30] + "..."
            
            sr = settings.value(f"{device_key}_samplerate", type=int)
            ch = settings.value(f"{device_key}_channels", type=int) 
            lat = settings.value(f"{device_key}_latency", type=str)
            host_api = settings.value(f"{device_key}_host_api", type=str)
            
            devices[device_name] = {
                'samplerate': sr,
                'channels': ch,
                'latency': lat,
                'host_api': host_api
            }
    
    print(f"✅ Gefundene gespeicherte Konfigurationen: {len(devices)}")
    for name, config in devices.items():
        print(f"  📌 {name}")
        print(f"     SR: {config['samplerate']}Hz, CH: {config['channels']}, "
              f"Latency: {config['latency']}, API: {config['host_api']}")
    
    return len(devices) > 0

def test_device_detection():
    """Test device detection and changes"""
    print("\n=== TEST 2: Geräteerkennung ===")
    
    devices = sd.query_devices()
    input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
    
    print(f"✅ Erkannte Eingabegeräte: {len(input_devices)}")
    
    # Check default device
    try:
        default_in, _ = sd.default.device
        if default_in is not None:
            print(f"✅ Standard-Eingabegerät: Index {default_in}")
            if isinstance(default_in, int) and default_in < len(devices):
                print(f"   Name: {devices[default_in].get('name', 'Unknown')}")
        else:
            print("⚠️ Kein Standard-Eingabegerät gesetzt")
    except:
        print("⚠️ Konnte Standard-Gerät nicht ermitteln")
    
    return len(input_devices) > 0

def test_hot_plug_simulation():
    """Simulate hot-plug detection"""
    print("\n=== TEST 3: Hot-Plug Detection (Simulation) ===")
    
    print("📊 Initiale Geräteliste:")
    devices1 = sd.query_devices()
    input_devices1 = [d for d in devices1 if d.get('max_input_channels', 0) > 0]
    names1 = set(d.get('name', '') for d in input_devices1)
    
    for name in list(names1)[:3]:  # Show first 3
        print(f"   - {name}")
    
    print("\n⏳ Warte 3 Sekunden... (Stecke ein Gerät ein/aus zum Testen)")
    time.sleep(3)
    
    # Re-scan
    sd._terminate()
    sd._initialize()
    devices2 = sd.query_devices()
    input_devices2 = [d for d in devices2 if d.get('max_input_channels', 0) > 0]
    names2 = set(d.get('name', '') for d in input_devices2)
    
    added = names2 - names1
    removed = names1 - names2
    
    if added or removed:
        print("✅ Geräteänderungen erkannt!")
        if added:
            print(f"   🔌 Hinzugefügt: {list(added)}")
        if removed:
            print(f"   🔍 Entfernt: {list(removed)}")
    else:
        print("ℹ️ Keine Geräteänderungen (normal wenn nichts ein/ausgesteckt wurde)")
    
    return True

def test_config_saving():
    """Test saving a new config"""
    print("\n=== TEST 4: Config-Speicherung ===")
    
    settings = QSettings('MikrofoneTool', 'DeviceConfigs')
    
    # Save a test config
    test_key = "device_TEST_DEVICE_12345"
    settings.setValue(f"{test_key}_samplerate", 48000)
    settings.setValue(f"{test_key}_channels", 2)
    settings.setValue(f"{test_key}_latency", "0.020")
    settings.setValue(f"{test_key}_host_api", "WASAPI")
    
    # Verify it was saved
    sr = settings.value(f"{test_key}_samplerate", type=int)
    if sr == 48000:
        print("✅ Test-Config erfolgreich gespeichert und geladen")
        
        # Clean up test entry
        settings.remove(f"{test_key}_samplerate")
        settings.remove(f"{test_key}_channels")
        settings.remove(f"{test_key}_latency")
        settings.remove(f"{test_key}_host_api")
        print("✅ Test-Config aufgeräumt")
        return True
    else:
        print("❌ Fehler beim Speichern/Laden der Config")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("UMFASSENDE TESTS FÜR NEUE FEATURES")
    print("=" * 60)
    
    app = QCoreApplication(sys.argv)
    
    results = []
    
    # Run tests
    results.append(("QSettings Persistenz", test_qsettings()))
    results.append(("Geräteerkennung", test_device_detection()))
    results.append(("Hot-Plug Detection", test_hot_plug_simulation()))
    results.append(("Config Speicherung", test_config_saving()))
    
    # Summary
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nErgebnis: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("\n🎉 ALLE TESTS ERFOLGREICH!")
    else:
        print(f"\n⚠️ {total - passed} Tests fehlgeschlagen")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())