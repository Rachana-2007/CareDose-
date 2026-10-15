"""
Medicine Guardian - Alarm System Module
Handles recurring 6-minute medicine reminder alarm with pleasant 60% volume gentle chime alert.
"""

import time
import threading
import sys
import math
import struct
import wave
import io

# Platform specific sound / audio playback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlarmSystem:
    def __init__(self, interval_seconds: int = 120, volume: float = 0.60):
        """
        Initialize the Alarm System.
        :param interval_seconds: Time between alarms (default: 120 seconds = 2 minutes)
        :param volume: Alarm audio volume scale (default: 0.60 = 60% volume)
        """
        self.interval_seconds = interval_seconds
        self.volume = max(0.0, min(1.0, volume)) # Clamp volume between 0% and 100%
        self.is_running = False
        self.is_alarm_active = False
        self._thread = None
        self._alarm_sound_thread = None
        self.on_alarm_triggered_callback = None
        
        # Pre-generate soft 60% volume chime WAV audio data
        self.chime_wav_bytes = self._generate_soft_chime_wav(volume=self.volume)

    def _generate_soft_chime_wav(self, volume: float = 0.60, duration_sec: float = 1.0) -> bytes:
        """
        Generate a gentle, pleasant chime audio byte array at exactly 40% volume amplitude.
        Prevents harsh loud beeps by using smooth harmonic sine waves with exponential decay.
        """
        sample_rate = 44100
        # 16-bit max PCM amplitude is 32767. Scale amplitude by volume (e.g. 40%)
        max_amplitude = int(32767 * volume)
        
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1) # Mono
            wf.setsampwidth(2) # 16-bit PCM
            wf.setframerate(sample_rate)
            
            samples = bytearray()
            n_samples = int(sample_rate * duration_sec)
            
            for i in range(n_samples):
                t = i / sample_rate
                # Smooth envelope (fade-in and exponential decay) to eliminate clicks
                envelope = math.exp(-2.5 * t) * (1 - math.exp(-25 * t))
                
                # Soft musical chord (C5 523Hz + E5 659Hz gentle harmony)
                freq1 = 523.25
                freq2 = 659.25
                val = (0.65 * math.sin(2 * math.pi * freq1 * t) + 0.35 * math.sin(2 * math.pi * freq2 * t))
                
                sample = int(max_amplitude * val * envelope)
                sample = max(-32768, min(32767, sample))
                samples.extend(struct.pack('<h', sample))
                
            wf.writeframes(samples)
            
        return buf.getvalue()

    def start(self, callback=None):
        """Start the recurring timer loop."""
        self.on_alarm_triggered_callback = callback
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[ALARM SYSTEM] Initialized. Alarm scheduled every {self.interval_seconds // 60} minutes ({self.interval_seconds}s) at {int(self.volume * 100)}% volume.")

    def _run_loop(self):
        """Background thread loop for 6-minute intervals."""
        next_alarm_time = time.time() + self.interval_seconds
        
        while self.is_running:
            time.sleep(1)
            remaining = int(next_alarm_time - time.time())
            
            if remaining <= 0 and not self.is_alarm_active:
                self.trigger_alarm()
                next_alarm_time = time.time() + self.interval_seconds

    def trigger_alarm(self):
        """Trigger the gentle 40% volume alarm sound and notify listeners."""
        if self.is_alarm_active:
            return
        
        self.is_alarm_active = True
        print("\n" + "=" * 55)
        print(f"[ALARM TRIGGERED] TIME TO TAKE YOUR MEDICINE! ({int(self.volume * 100)}% Volume)")
        print("   Phone access is LOCKED until medicine intake is verified by CV.")
        print("=" * 55 + "\n")

        # Launch continuous gentle alarm sound in separate thread
        self._alarm_sound_thread = threading.Thread(target=self._play_alarm_sound, daemon=True)
        self._alarm_sound_thread.start()

        if self.on_alarm_triggered_callback:
            self.on_alarm_triggered_callback()

    def _play_alarm_sound(self):
        """Play soft 60% volume chime audio repeatedly until alarm is silenced."""
        while self.is_alarm_active and self.is_running:
            if HAS_WINSOUND and self.chime_wav_bytes:
                try:
                    # Play gentle 60% volume chime audio synchronously
                    winsound.PlaySound(self.chime_wav_bytes, winsound.SND_MEMORY)
                except Exception:
                    time.sleep(1.0)
            else:
                sys.stdout.write('\a')
                sys.stdout.flush()
                time.sleep(1.0)
            
            # Short pause between chime repetitions
            time.sleep(0.8)

    def stop_alarm(self):
        """Silence the alarm after medicine verification."""
        if self.is_alarm_active:
            self.is_alarm_active = False
            # Stop any currently playing audio stream immediately
            if HAS_WINSOUND:
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass
            print("[ALARM SILENCED] Medicine intake confirmed! Alarm turned off.")

    def shutdown(self):
        """Shutdown alarm thread."""
        self.is_running = False
        self.stop_alarm()


if __name__ == "__main__":
    print("Testing Alarm System at 40% volume (10s test interval)...")
    alarm = AlarmSystem(interval_seconds=10, volume=0.40)
    alarm.start(callback=lambda: print("Callback fired!"))
    time.sleep(12)
    alarm.stop_alarm()
    alarm.shutdown()
