#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test with proper warmup - open stream BEFORE countdown"""

import pyaudio
import struct
import time

def test_with_warmup(device_index=1):
    """Test with stream opened before countdown"""
    
    p = pyaudio.PyAudio()
    
    RATE = 44100
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    print("Opening audio stream...")
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )
    
    print("Stream opened. Starting warmup...")
    
    # CRITICAL: Read and discard some data to activate the device
    print("Activating device (reading initial chunks)...")
    for i in range(10):  # Read 10 chunks to activate
        try:
            _ = stream.read(CHUNK, exception_on_overflow=False)
            print(f"  Warmup chunk {i+1}/10")
            time.sleep(0.05)
        except:
            pass
    
    print("\nDevice should be active now. Starting countdown...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Recording in {i}...")
        # Keep reading during countdown to keep device active
        try:
            _ = stream.read(CHUNK, exception_on_overflow=False)
        except:
            pass
        time.sleep(1)
    
    print("\n[RECORDING NOW - SPEAK!]")
    
    # Record for real
    frames = []
    duration = 3
    num_chunks = int(RATE / CHUNK * duration)
    
    for i in range(num_chunks):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Check amplitude every 20 chunks
            if i % 20 == 0:
                samples = struct.unpack('h' * (len(data) // 2), data)
                if samples:
                    max_amp = max(abs(s) / 32768.0 for s in samples)
                    print(f"  Chunk {i}/{num_chunks}: max_amp={max_amp:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Analyze
    all_data = b''.join(frames)
    print(f"\n[COMPLETE] Recorded {len(all_data)} bytes")
    
    samples = struct.unpack('h' * (len(all_data) // 2), all_data)
    if samples:
        max_amp = max(abs(s) / 32768.0 for s in samples)
        avg_amp = sum(abs(s) for s in samples) / len(samples) / 32768.0
        print(f"Results: max_amp={max_amp:.4f}, avg_amp={avg_amp:.6f}")
        
        if max_amp < 0.001:
            print("[WARNING] Near zero audio!")
        elif max_amp < 0.01:
            print("[WARNING] Low audio level")
        else:
            print("[OK] Good audio level!")

if __name__ == "__main__":
    print("Testing Device 1 with proper warmup sequence")
    print("The device will be activated BEFORE the countdown\n")
    test_with_warmup(device_index=1)