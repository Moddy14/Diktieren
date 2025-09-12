#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare PyAudio direct vs speech_recognition audio capture"""

import pyaudio
import speech_recognition as sr
import struct
import time
import numpy as np

def test_pyaudio_direct(device_index=1, duration=2):
    """Test with PyAudio directly"""
    print("\n=== PYAUDIO DIRECT TEST ===")
    
    p = pyaudio.PyAudio()
    
    RATE = 44100
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
    
    print("Recording with PyAudio...")
    frames = []
    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    all_data = b''.join(frames)
    samples = struct.unpack('h' * (len(all_data) // 2), all_data)
    max_amp = max(abs(s) / 32768.0 for s in samples)
    avg_amp = sum(abs(s) for s in samples) / len(samples) / 32768.0
    
    print(f"PyAudio Results:")
    print(f"  Bytes: {len(all_data)}")
    print(f"  Max amplitude: {max_amp:.4f}")
    print(f"  Avg amplitude: {avg_amp:.6f}")
    
    return all_data

def test_speech_recognition(device_index=1, duration=2):
    """Test with speech_recognition"""
    print("\n=== SPEECH_RECOGNITION TEST ===")
    
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 100
    recognizer.dynamic_energy_threshold = False
    
    mic = sr.Microphone(
        device_index=device_index,
        sample_rate=44100,
        chunk_size=512
    )
    
    with mic as source:
        print("Recording with speech_recognition...")
        audio = recognizer.record(source, duration=duration)
        raw_data = audio.get_raw_data()
        
        samples = struct.unpack('h' * (len(raw_data) // 2), raw_data)
        max_amp = max(abs(s) / 32768.0 for s in samples)
        avg_amp = sum(abs(s) for s in samples) / len(samples) / 32768.0
        
        print(f"Speech Recognition Results:")
        print(f"  Bytes: {len(raw_data)}")
        print(f"  Max amplitude: {max_amp:.4f}")
        print(f"  Avg amplitude: {avg_amp:.6f}")
        
        # Check the actual PyAudio stream inside speech_recognition
        print(f"\nInternal stream info:")
        print(f"  Stream type: {type(source.stream)}")
        if hasattr(source.stream, 'pyaudio_stream'):
            pa_stream = source.stream.pyaudio_stream
            print(f"  PyAudio stream: {pa_stream}")
            if hasattr(pa_stream, 'get_input_latency'):
                print(f"  Input latency: {pa_stream.get_input_latency()}")
        
        return raw_data

def compare_audio(data1, data2):
    """Compare two audio buffers"""
    print("\n=== COMPARISON ===")
    
    samples1 = struct.unpack('h' * (len(data1) // 2), data1)
    samples2 = struct.unpack('h' * (len(data2) // 2), data2)
    
    # Compare first 100 samples
    print("\nFirst 10 samples comparison:")
    print("PyAudio:  ", [f"{s:6d}" for s in samples1[:10]])
    print("SpeechRec:", [f"{s:6d}" for s in samples2[:10]])
    
    # Check if data is identical
    if data1[:1000] == data2[:1000]:
        print("\n[INFO] First 1000 bytes are IDENTICAL")
    else:
        print("\n[WARNING] Data differs between methods!")
        
        # Find first difference
        for i in range(min(len(data1), len(data2))):
            if data1[i] != data2[i]:
                print(f"First difference at byte {i}: PyAudio={data1[i]}, SR={data2[i]}")
                break

if __name__ == "__main__":
    print("Testing Device 1 audio capture comparison")
    print("Please speak during the tests!\n")
    
    # Warmup
    print("Warmup (0.5s)...")
    time.sleep(0.5)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("\n[SPEAK NOW!]")
    
    # Test both methods
    pyaudio_data = test_pyaudio_direct(device_index=1, duration=2)
    
    print("\n--- Switching to speech_recognition ---")
    time.sleep(1)
    
    sr_data = test_speech_recognition(device_index=1, duration=2)
    
    # Compare results
    compare_audio(pyaudio_data, sr_data)