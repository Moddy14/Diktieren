#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test speech_recognition with Device 1"""

import speech_recognition as sr
import struct
import time

def test_sr_device1():
    """Test speech_recognition with Device 1"""
    
    device_index = 1
    sample_rate = 44100
    chunk_size = 512
    
    print(f"Testing Device {device_index} with speech_recognition")
    print(f"Parameters: {sample_rate}Hz, chunk={chunk_size}")
    
    # Create recognizer
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 100  # Low threshold
    recognizer.dynamic_energy_threshold = False
    
    # Warmup
    print("\nWarmup phase (0.5s)...")
    time.sleep(0.5)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("\n[RECORDING] - SPEAK NOW!")
    
    try:
        # Open microphone
        mic = sr.Microphone(
            device_index=device_index,
            sample_rate=sample_rate,
            chunk_size=chunk_size
        )
        
        with mic as source:
            print(f"Microphone opened successfully")
            print(f"Stream type: {type(source.stream)}")
            print(f"Stream active: {source.stream.is_active() if hasattr(source.stream, 'is_active') else 'unknown'}")
            
            # Test 1: Use recognizer.record() for direct recording
            print("\nTest 1: Using recognizer.record() for 3 seconds...")
            audio = recognizer.record(source, duration=3.0)
            raw_data = audio.get_raw_data()
            print(f"Recorded {len(raw_data)} bytes")
            
            # Analyze audio
            samples = struct.unpack('h' * (len(raw_data) // 2), raw_data)
            if samples:
                max_amp = max(abs(s) / 32768.0 for s in samples)
                avg_amp = sum(abs(s) for s in samples[:10000]) / min(10000, len(samples)) / 32768.0
                print(f"Audio analysis: max_amp={max_amp:.4f}, avg_amp={avg_amp:.6f}")
                
                if max_amp < 0.001:
                    print("[WARNING] Audio near zero!")
                elif max_amp < 0.01:
                    print("[WARNING] Audio level low")
                else:
                    print("[OK] Audio level good")
            
            # Test 2: Try speech recognition
            print("\nTest 2: Attempting speech recognition...")
            try:
                text = recognizer.recognize_google(audio, language="de-DE")
                print(f"[SUCCESS] Recognized: {text}")
            except sr.UnknownValueError:
                print("[INFO] No speech detected")
            except sr.RequestError as e:
                print(f"[ERROR] Recognition error: {e}")
            
            # Test 3: Try listen() method with timeout
            print("\nTest 3: Using recognizer.listen() with timeout=5...")
            try:
                audio2 = recognizer.listen(source, timeout=5.0)
                raw_data2 = audio2.get_raw_data()
                print(f"Listen captured {len(raw_data2)} bytes")
                
                samples2 = struct.unpack('h' * (len(raw_data2) // 2), raw_data2)
                if samples2:
                    max_amp2 = max(abs(s) / 32768.0 for s in samples2)
                    print(f"Listen audio: max_amp={max_amp2:.4f}")
                    
            except sr.WaitTimeoutError:
                print("[INFO] Listen timeout - no speech detected above threshold")
            
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test_sr_device1()