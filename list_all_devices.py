#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List all audio devices with detailed info"""

import pyaudio
import sounddevice as sd

def list_pyaudio_devices():
    """List all PyAudio devices"""
    print("\n=== PYAUDIO DEVICES ===")
    p = pyaudio.PyAudio()
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"\nDevice {i}: {info['name']}")
            print(f"  Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
            print(f"  Max Input Channels: {info['maxInputChannels']}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']}")
            
            # Check if it's the Bluetooth device
            if "Buds3" in info['name'] or "Galaxy" in info['name']:
                print("  >>> THIS IS THE BLUETOOTH DEVICE <<<")
    
    p.terminate()

def list_sounddevice_devices():
    """List all sounddevice devices"""
    print("\n=== SOUNDDEVICE DEVICES ===")
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"\nDevice {i}: {device['name']}")
            print(f"  Host API: {sd.query_hostapis(device['hostapi'])['name']}")
            print(f"  Max Input Channels: {device['max_input_channels']}")
            print(f"  Default Sample Rate: {device['default_samplerate']}")
            
            # Check if it's the Bluetooth device
            if "Buds3" in device['name'] or "Galaxy" in device['name']:
                print("  >>> THIS IS THE BLUETOOTH DEVICE <<<")

def test_specific_device(device_index=1):
    """Test a specific device"""
    print(f"\n=== TESTING DEVICE {device_index} ===")
    
    p = pyaudio.PyAudio()
    info = p.get_device_info_by_index(device_index)
    
    print(f"Device: {info['name']}")
    print(f"Testing different sample rates...")
    
    # Test different sample rates
    test_rates = [8000, 16000, 22050, 44100, 48000]
    
    for rate in test_rates:
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=512,
                start=False  # Don't start immediately
            )
            stream.close()
            print(f"  {rate}Hz: OK")
        except Exception as e:
            print(f"  {rate}Hz: FAILED - {e}")
    
    p.terminate()

if __name__ == "__main__":
    list_pyaudio_devices()
    list_sounddevice_devices()
    test_specific_device(1)