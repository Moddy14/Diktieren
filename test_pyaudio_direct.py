#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct PyAudio test for Device 1 (Samsung Galaxy Buds3 Pro)"""

import pyaudio
import struct
import time

def test_device(device_index=1, duration=3):
    """Test audio capture from specific device"""
    
    p = pyaudio.PyAudio()
    
    # Get device info
    info = p.get_device_info_by_index(device_index)
    print(f"\nTesting Device {device_index}: {info['name']}")
    print(f"Max input channels: {info['maxInputChannels']}")
    print(f"Default sample rate: {info['defaultSampleRate']}")
    
    # Test parameters from working microphone test
    RATE = 44100
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    print(f"\nUsing: {RATE}Hz, {CHUNK} chunk size, {CHANNELS} channel(s)")
    
    # Warmup
    print("Warmup phase (0.5s)...")
    time.sleep(0.5)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("\n[RECORDING] - SPEAK NOW!")
    
    try:
        # Open stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print(f"Stream opened successfully!")
        print(f"Stream active: {stream.is_active()}")
        
        frames = []
        num_chunks = int(RATE / CHUNK * duration)
        
        for i in range(num_chunks):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Analyze every 10th chunk
                if i % 10 == 0:
                    samples = struct.unpack('h' * (len(data) // 2), data)
                    if samples:
                        max_amp = max(abs(s) / 32768.0 for s in samples)
                        print(f"  Chunk {i}/{num_chunks}: max_amp={max_amp:.4f}")
            except Exception as e:
                print(f"Error reading chunk {i}: {e}")
        
        stream.stop_stream()
        stream.close()
        
        # Analyze full recording
        all_data = b''.join(frames)
        print(f"\n[COMPLETE] Recording complete: {len(all_data)} bytes")
        
        all_samples = struct.unpack('h' * (len(all_data) // 2), all_data)
        if all_samples:
            max_amp = max(abs(s) / 32768.0 for s in all_samples)
            avg_amp = sum(abs(s) for s in all_samples) / len(all_samples) / 32768.0
            print(f"Overall: max_amp={max_amp:.4f}, avg_amp={avg_amp:.6f}")
            
            if max_amp < 0.001:
                print("[WARNING] Audio amplitude near zero - microphone may not be active!")
            elif max_amp < 0.01:
                print("[WARNING] Audio level very low - speak louder or check microphone")
            else:
                print("[OK] Audio level looks good!")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    test_device(device_index=1, duration=3)