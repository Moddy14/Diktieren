#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Device 9 (DirectSound Bluetooth)"""

import pyaudio
import struct
import time

def test_device(device_index=9):
    """Test audio capture from Device 9"""
    
    p = pyaudio.PyAudio()
    
    # Get device info
    info = p.get_device_info_by_index(device_index)
    print(f"\nTesting Device {device_index}: {info['name']}")
    print(f"Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
    print(f"Max input channels: {info['maxInputChannels']}")
    print(f"Default sample rate: {info['defaultSampleRate']}")
    
    RATE = 44100
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    print(f"\nUsing: {RATE}Hz, {CHUNK} chunk size, {CHANNELS} channel(s)")
    
    print("\nOpening stream...")
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print("Stream opened! Warming up...")
        
        # Warmup reads
        for i in range(5):
            _ = stream.read(CHUNK, exception_on_overflow=False)
            time.sleep(0.1)
        
        print("\n[RECORDING] - SPEAK NOW!")
        
        frames = []
        duration = 3
        num_chunks = int(RATE / CHUNK * duration)
        
        for i in range(num_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            if i % 30 == 0:
                samples = struct.unpack('h' * (len(data) // 2), data)
                if samples:
                    max_amp = max(abs(s) / 32768.0 for s in samples)
                    print(f"  Chunk {i}/{num_chunks}: max_amp={max_amp:.4f}")
        
        stream.stop_stream()
        stream.close()
        
        # Analyze
        all_data = b''.join(frames)
        print(f"\n[COMPLETE] Recorded {len(all_data)} bytes")
        
        all_samples = struct.unpack('h' * (len(all_data) // 2), all_data)
        if all_samples:
            max_sample = max(abs(s) for s in all_samples)
            max_amp = max_sample / 32768.0
            avg_amp = sum(abs(s) for s in all_samples) / len(all_samples) / 32768.0
            print(f"Overall: max_sample={max_sample}, max_amp={max_amp:.4f}, avg_amp={avg_amp:.6f}")
            
            # Show distribution
            ranges = [0, 10, 100, 1000, 10000, 32768]
            for i in range(len(ranges)-1):
                count = sum(1 for s in all_samples if ranges[i] <= abs(s) < ranges[i+1])
                if count > 0:
                    print(f"  Samples in range [{ranges[i]:5d}-{ranges[i+1]:5d}): {count}")
            
            if max_amp > 0.01:
                print("[OK] Good audio level!")
                print("\n>>> DEVICE 9 WORKS BETTER! <<<")
            elif max_amp > 0.001:
                print("[WARNING] Low audio level")
            else:
                print("[WARNING] Near zero audio!")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    print("Testing Device 9 (DirectSound API)")
    test_device(device_index=9)