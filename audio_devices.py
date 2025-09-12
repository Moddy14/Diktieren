# -*- coding: utf-8 -*-
"""
Erweiterte Audio-Geräteverwaltung mit sounddevice.
Robuste Funktionen für Standard-Mikrofon Ermittlung und Konfiguration.
"""
# pip install sounddevice numpy
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import sounddevice as sd
import numpy as np
import time
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AudioAPI(Enum):
    """Unterstützte Audio-APIs."""
    MME = "MME"
    DIRECTSOUND = "Windows DirectSound"
    WASAPI = "Windows WASAPI"
    WDM_KS = "Windows WDM-KS"
    ASIO = "ASIO"
    CORE_AUDIO = "Core Audio"
    ALSA = "ALSA"
    JACK = "JACK"
    PULSE = "PulseAudio"

@dataclass
class DeviceInfo:
    """Strukturierte Geräteinformationen."""
    index: int
    name: str
    hostapi: str
    hostapi_index: int
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    default_low_input_latency: float
    default_high_input_latency: float
    is_default_input: bool = False
    is_default_output: bool = False
    
    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0
    
    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0
    
    @property
    def is_bluetooth(self) -> bool:
        """Heuristik für Bluetooth-Geräte."""
        bt_keywords = ['bluetooth', 'buds', 'airpods', 'headset', 'hands-free']
        name_lower = self.name.lower()
        return any(kw in name_lower for kw in bt_keywords)
    
    def __str__(self) -> str:
        flags = []
        if self.is_default_input:
            flags.append("DEFAULT INPUT")
        if self.is_default_output:
            flags.append("DEFAULT OUTPUT")
        if self.is_bluetooth:
            flags.append("BT")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        
        return (f"[{self.index}] {self.name} ({self.hostapi}){flag_str}\n"
                f"    Channels: IN={self.max_input_channels} OUT={self.max_output_channels} | "
                f"SR={self.default_samplerate:.0f}Hz | "
                f"Latency={self.default_low_input_latency*1000:.1f}ms")

class AudioDeviceManager:
    """Erweiterte Verwaltung von Audio-Geräten."""
    
    def __init__(self):
        self._cache_time: Optional[float] = None
        self._cached_devices: Optional[List[DeviceInfo]] = None
        self._cache_duration = 2.0  # Cache für 2 Sekunden
        
    def list_all_devices(self, force_refresh: bool = False) -> List[DeviceInfo]:
        """Alle Audio-Geräte mit detaillierten Infos auflisten."""
        if not force_refresh and self._cached_devices and self._cache_time:
            if time.time() - self._cache_time < self._cache_duration:
                return self._cached_devices
        
        devices = []
        try:
            devs = sd.query_devices()
            apis = sd.query_hostapis()
            default_input, default_output = self._get_default_indices()
            
            for idx, d in enumerate(devs):
                hostapi_idx = d.get("hostapi", 0)
                hostapi_name = apis[hostapi_idx]["name"] if hostapi_idx < len(apis) else "Unknown"
                
                device = DeviceInfo(
                    index=idx,
                    name=d.get("name", f"Device {idx}"),
                    hostapi=hostapi_name,
                    hostapi_index=hostapi_idx,
                    max_input_channels=d.get("max_input_channels", 0),
                    max_output_channels=d.get("max_output_channels", 0),
                    default_samplerate=d.get("default_samplerate", 44100),
                    default_low_input_latency=d.get("default_low_input_latency", 0.01),
                    default_high_input_latency=d.get("default_high_input_latency", 0.1),
                    is_default_input=(idx == default_input),
                    is_default_output=(idx == default_output)
                )
                devices.append(device)
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Geräteliste: {e}")
            
        self._cached_devices = devices
        self._cache_time = time.time()
        return devices
    
    def list_input_devices(self, api_filter: Optional[AudioAPI] = None) -> List[DeviceInfo]:
        """Nur Eingabegeräte auflisten, optional nach API gefiltert."""
        devices = self.list_all_devices()
        result = [d for d in devices if d.is_input]
        
        if api_filter:
            result = [d for d in result if api_filter.value in d.hostapi]
            
        return result
    
    def list_output_devices(self, api_filter: Optional[AudioAPI] = None) -> List[DeviceInfo]:
        """Nur Ausgabegeräte auflisten, optional nach API gefiltert."""
        devices = self.list_all_devices()
        result = [d for d in devices if d.is_output]
        
        if api_filter:
            result = [d for d in result if api_filter.value in d.hostapi]
            
        return result
    
    def find_devices_by_name(self, search_term: str, 
                           only_inputs: bool = False,
                           only_outputs: bool = False,
                           exact_match: bool = False) -> List[DeviceInfo]:
        """
        Geräte nach Namen suchen.
        
        Args:
            search_term: Suchbegriff
            only_inputs: Nur Eingabegeräte
            only_outputs: Nur Ausgabegeräte  
            exact_match: Exakte Übereinstimmung statt Teilstring
        """
        devices = self.list_all_devices()
        search_lower = search_term.lower()
        
        results = []
        for d in devices:
            if only_inputs and not d.is_input:
                continue
            if only_outputs and not d.is_output:
                continue
                
            if exact_match:
                if d.name.lower() == search_lower:
                    results.append(d)
            else:
                if search_lower in d.name.lower():
                    results.append(d)
                    
        return results
    
    def _get_default_indices(self) -> Tuple[Optional[int], Optional[int]]:
        """Aktuelle Default-Geräte-Indices ermitteln."""
        try:
            din, dout = sd.default.device
            
            # Konvertiere zu Index falls Name
            if isinstance(din, str):
                matches = self.find_devices_by_name(din, only_inputs=True)
                din = matches[0].index if matches else None
            elif not isinstance(din, int) or din < 0:
                din = None
                
            if isinstance(dout, str):
                matches = self.find_devices_by_name(dout, only_outputs=True)
                dout = matches[0].index if matches else None
            elif not isinstance(dout, int) or dout < 0:
                dout = None
                
            return din, dout
        except:
            return None, None
    
    def get_default_input(self) -> Optional[DeviceInfo]:
        """Aktuelles Standard-Eingabegerät ermitteln."""
        din, _ = self._get_default_indices()
        if din is not None:
            devices = self.list_all_devices()
            for d in devices:
                if d.index == din:
                    return d
        return None
    
    def get_default_output(self) -> Optional[DeviceInfo]:
        """Aktuelles Standard-Ausgabegerät ermitteln."""
        _, dout = self._get_default_indices()
        if dout is not None:
            devices = self.list_all_devices()
            for d in devices:
                if d.index == dout:
                    return d
        return None
    
    def set_default_input(self, device: int | str | DeviceInfo | None) -> DeviceInfo:
        """
        Standard-Eingabegerät setzen.
        
        Args:
            device: Index, Name, DeviceInfo-Objekt oder None für System-Default
            
        Returns:
            Das gesetzte Gerät
        """
        _, dout = self._get_default_indices()
        
        if device is None:
            sd.default.device = (None, dout)
            logger.info("Eingabegerät auf System-Default zurückgesetzt")
            return self.get_default_input()
        
        if isinstance(device, DeviceInfo):
            device = device.index
            
        if isinstance(device, str):
            matches = self.find_devices_by_name(device, only_inputs=True)
            if not matches:
                raise ValueError(f"Kein Eingabegerät mit Namen '{device}' gefunden")
            if len(matches) > 1:
                logger.warning(f"Mehrere Treffer für '{device}', verwende ersten: {matches[0].name}")
            device = matches[0].index
            
        # Validierung
        devices = self.list_all_devices()
        target = None
        for d in devices:
            if d.index == device:
                target = d
                break
                
        if not target:
            raise ValueError(f"Gerät mit Index {device} nicht gefunden")
        if not target.is_input:
            raise ValueError(f"Gerät {target.name} ist kein Eingabegerät")
            
        sd.default.device = (device, dout)
        logger.info(f"Standard-Eingabegerät gesetzt: {target.name} (Index {device})")
        return target
    
    def set_default_output(self, device: int | str | DeviceInfo | None) -> DeviceInfo:
        """
        Standard-Ausgabegerät setzen.
        
        Args:
            device: Index, Name, DeviceInfo-Objekt oder None für System-Default
            
        Returns:
            Das gesetzte Gerät
        """
        din, _ = self._get_default_indices()
        
        if device is None:
            sd.default.device = (din, None)
            logger.info("Ausgabegerät auf System-Default zurückgesetzt")
            return self.get_default_output()
        
        if isinstance(device, DeviceInfo):
            device = device.index
            
        if isinstance(device, str):
            matches = self.find_devices_by_name(device, only_outputs=True)
            if not matches:
                raise ValueError(f"Kein Ausgabegerät mit Namen '{device}' gefunden")
            if len(matches) > 1:
                logger.warning(f"Mehrere Treffer für '{device}', verwende ersten: {matches[0].name}")
            device = matches[0].index
            
        # Validierung
        devices = self.list_all_devices()
        target = None
        for d in devices:
            if d.index == device:
                target = d
                break
                
        if not target:
            raise ValueError(f"Gerät mit Index {device} nicht gefunden")
        if not target.is_output:
            raise ValueError(f"Gerät {target.name} ist kein Ausgabegerät")
            
        sd.default.device = (din, device)
        logger.info(f"Standard-Ausgabegerät gesetzt: {target.name} (Index {device})")
        return target
    
    def test_device(self, device: int | str | DeviceInfo,
                   duration_ms: int = 500,
                   samplerate: Optional[int] = None) -> Dict[str, Any]:
        """
        Testet ein Eingabegerät durch kurze Aufnahme.
        
        Args:
            device: Zu testendes Gerät
            duration_ms: Testdauer in Millisekunden
            samplerate: Sampling-Rate (None = Gerät-Default)
            
        Returns:
            Dict mit Test-Ergebnissen (success, max_level, rms, error)
        """
        if isinstance(device, str):
            matches = self.find_devices_by_name(device, only_inputs=True)
            if not matches:
                return {"success": False, "error": f"Gerät '{device}' nicht gefunden"}
            device = matches[0]
        elif isinstance(device, int):
            devices = self.list_all_devices()
            device = next((d for d in devices if d.index == device), None)
            if not device:
                return {"success": False, "error": f"Gerät Index {device} nicht gefunden"}
                
        if not isinstance(device, DeviceInfo):
            return {"success": False, "error": "Ungültiger Gerätetyp"}
            
        if not device.is_input:
            return {"success": False, "error": f"{device.name} ist kein Eingabegerät"}
            
        sr = samplerate or int(device.default_samplerate)
        duration = duration_ms / 1000.0
        
        try:
            # Aufnahme mit Timeout
            recording = sd.rec(
                int(sr * duration),
                samplerate=sr,
                channels=min(2, device.max_input_channels),
                device=device.index,
                dtype='float32'
            )
            sd.wait()
            
            # Audio-Analyse
            max_level = float(np.max(np.abs(recording)))
            rms = float(np.sqrt(np.mean(np.square(recording))))
            
            result = {
                "success": True,
                "device_name": device.name,
                "device_index": device.index,
                "samplerate": sr,
                "channels": min(2, device.max_input_channels),
                "max_level": max_level,
                "rms": rms,
                "has_signal": max_level > 0.001,
                "error": None
            }
            
            logger.info(f"Gerät {device.name} erfolgreich getestet: "
                       f"Max={max_level:.4f}, RMS={rms:.4f}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Fehler beim Testen von {device.name}: {error_msg}")
            return {
                "success": False,
                "device_name": device.name,
                "device_index": device.index,
                "error": error_msg
            }
    
    def find_best_input_device(self, prefer_bluetooth: bool = False,
                              exclude_virtual: bool = True) -> Optional[DeviceInfo]:
        """
        Findet das beste verfügbare Eingabegerät basierend auf Heuristiken.
        
        Args:
            prefer_bluetooth: Bluetooth-Geräte bevorzugen
            exclude_virtual: Virtuelle Geräte ausschließen
            
        Returns:
            Das beste gefundene Gerät oder None
        """
        devices = self.list_input_devices()
        
        if not devices:
            return None
            
        # Virtuelle Geräte filtern
        if exclude_virtual:
            virtual_keywords = ['virtual', 'cable', 'loopback', 'vb-audio', 'voicemeeter']
            devices = [d for d in devices 
                      if not any(kw in d.name.lower() for kw in virtual_keywords)]
        
        # Scoring-System
        scored = []
        for d in devices:
            score = 0
            
            # Default-Gerät bevorzugen
            if d.is_default_input:
                score += 100
                
            # Bluetooth-Präferenz
            if d.is_bluetooth:
                score += 50 if prefer_bluetooth else -20
                
            # Host-API Präferenzen (Windows)
            if "WASAPI" in d.hostapi:
                score += 30
            elif "MME" in d.hostapi:
                score += 20
            elif "DirectSound" in d.hostapi:
                score += 10
                
            # Niedrige Latenz bevorzugen
            if d.default_low_input_latency < 0.01:
                score += 15
            elif d.default_low_input_latency < 0.02:
                score += 10
                
            # Standard-Samplerate bevorzugen
            if d.default_samplerate in [44100, 48000]:
                score += 5
                
            scored.append((score, d))
            
        # Nach Score sortieren
        scored.sort(key=lambda x: x[0], reverse=True)
        
        if scored:
            best = scored[0][1]
            logger.info(f"Bestes Eingabegerät: {best.name} (Score: {scored[0][0]})")
            return best
            
        return None
    
    def monitor_device_changes(self, callback, interval_ms: int = 1000):
        """
        Überwacht Geräteänderungen (vereinfachte Version ohne Threading).
        
        Args:
            callback: Funktion die bei Änderungen aufgerufen wird
            interval_ms: Check-Intervall in Millisekunden
        """
        import hashlib
        
        def get_device_hash():
            devices = self.list_all_devices(force_refresh=True)
            device_str = ''.join(f"{d.index}{d.name}" for d in devices)
            return hashlib.md5(device_str.encode()).hexdigest()
        
        last_hash = get_device_hash()
        
        def check():
            nonlocal last_hash
            current_hash = get_device_hash()
            if current_hash != last_hash:
                last_hash = current_hash
                callback(self.list_all_devices())
                
        # Für Qt-Integration würde man QTimer verwenden
        # Hier nur als Beispiel-Struktur
        return check


# Convenience-Funktionen für schnellen Zugriff
_manager = AudioDeviceManager()

def list_input_devices() -> List[DeviceInfo]:
    """Alle Eingabegeräte auflisten."""
    return _manager.list_input_devices()

def get_default_input() -> Optional[DeviceInfo]:
    """Aktuelles Standard-Eingabegerät."""
    return _manager.get_default_input()

def set_default_input(device: int | str | None) -> DeviceInfo:
    """Standard-Eingabegerät setzen."""
    return _manager.set_default_input(device)

def find_device(name: str) -> Optional[DeviceInfo]:
    """Erstes Gerät mit passendem Namen finden."""
    matches = _manager.find_devices_by_name(name)
    return matches[0] if matches else None

def test_device(device: int | str | DeviceInfo) -> Dict[str, Any]:
    """Eingabegerät testen."""
    return _manager.test_device(device)


if __name__ == "__main__":
    import sys
    
    # Logging einrichten
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    manager = AudioDeviceManager()
    
    print("=" * 60)
    print("AUDIO DEVICE MANAGER - Erweiterte Geräteverwaltung")
    print("=" * 60)
    
    # Alle Geräte anzeigen
    print("\n📋 ALLE AUDIO-GERÄTE:")
    print("-" * 60)
    for device in manager.list_all_devices():
        print(device)
    
    # Nur Eingabegeräte
    print("\n🎤 EINGABEGERÄTE:")
    print("-" * 60)
    for device in manager.list_input_devices():
        print(f"  {device.index}: {device.name} ({device.hostapi})")
    
    # Standard-Geräte
    print("\n⭐ STANDARD-GERÄTE:")
    print("-" * 60)
    default_in = manager.get_default_input()
    default_out = manager.get_default_output()
    print(f"  Input:  {default_in.name if default_in else 'Nicht gesetzt'}")
    print(f"  Output: {default_out.name if default_out else 'Nicht gesetzt'}")
    
    # Bestes Gerät finden
    print("\n🔍 BESTES EINGABEGERÄT:")
    print("-" * 60)
    best = manager.find_best_input_device()
    if best:
        print(f"  Empfehlung: {best.name}")
        print(f"  Begründung: {'Bluetooth' if best.is_bluetooth else 'Standard'}, "
              f"{best.hostapi}, {best.default_samplerate}Hz")
    
    # Interaktive Tests
    if len(sys.argv) > 1:
        search = ' '.join(sys.argv[1:])
        print(f"\n🔎 SUCHE NACH: '{search}'")
        print("-" * 60)
        matches = manager.find_devices_by_name(search)
        if matches:
            for m in matches:
                print(f"  Gefunden: {m.name} (Index {m.index})")
                
            if input("\n  Gerät testen? (j/n): ").lower() == 'j':
                print("\n  Starte 2-Sekunden-Test...")
                result = manager.test_device(matches[0])
                if result['success']:
                    print(f"  ✅ Test erfolgreich!")
                    print(f"     Max-Pegel: {result['max_level']:.4f}")
                    print(f"     RMS: {result['rms']:.4f}")
                    print(f"     Signal erkannt: {'Ja' if result['has_signal'] else 'Nein'}")
                else:
                    print(f"  ❌ Test fehlgeschlagen: {result['error']}")
                    
            if input("\n  Als Standard setzen? (j/n): ").lower() == 'j':
                try:
                    manager.set_default_input(matches[0])
                    print(f"  ✅ {matches[0].name} als Standard-Eingabe gesetzt")
                except Exception as e:
                    print(f"  ❌ Fehler: {e}")
        else:
            print(f"  Keine Geräte gefunden")
    
    print("\n" + "=" * 60)
    print("Tipp: Programmaufruf mit Gerätenamen zum Suchen/Testen")
    print("      python audio_devices.py Galaxy Buds")
    print("=" * 60)