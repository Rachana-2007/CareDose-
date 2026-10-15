"""
Medicine Guardian - Main Application Launcher
Orchestrates 6-minute recurring alarm, phone locking mechanism, and OpenCV tablet intake verification.
"""

import time
import argparse
import os
import sys

# Ensure local module directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from alarm_system import AlarmSystem
from phone_lock import PhoneLockManager
from cv_detector import MedicineIntakeDetectorCV


class MedicineGuardianApp:
    def __init__(self, alarm_interval_seconds: int = 120, camera_index: int = 0):
        """
        Initialize the complete Medicine Guardian App.
        :param alarm_interval_seconds: Time interval between alarms (default: 120s = 2 min)
        :param camera_index: Camera index for OpenCV webcams
        """
        self.alarm_interval_seconds = alarm_interval_seconds
        self.camera_index = camera_index
        
        # Instantiate modules
        self.alarm_system = AlarmSystem(interval_seconds=alarm_interval_seconds)
        self.phone_locker = PhoneLockManager(enforce_system_lock=False)
        self.cv_detector = MedicineIntakeDetectorCV(
            camera_index=camera_index,
            required_tablet_frames=15,
            required_swallow_frames=18
        )

    def on_alarm_triggered(self):
        """Callback fired when the 2-minute alarm triggers."""
        # 1. Lock phone
        self.phone_locker.lock(reason="2-Minute Medicine Time Alert")

        # 2. Launch OpenCV Detection window to track person & tablet intake
        print("[MAIN] Launching OpenCV camera monitor for medicine verification...")
        success = self.cv_detector.run_detection_window(
            on_success_callback=self.on_intake_verified
        )

        if not success:
            print("[MAIN] Intake was not verified. Phone remains LOCKED until intake is complete!")

    def on_intake_verified(self):
        """Callback fired when OpenCV confirms tablet intake."""
        print("[MAIN] Medicine intake successfully verified by OpenCV!")
        # 1. Stop vibration / ringing alarm
        self.alarm_system.stop_alarm()

        # 2. Unlock phone
        self.phone_locker.unlock()

    def start(self, trigger_immediately: bool = True):
        """Start the Medicine Guardian system."""
        print("=" * 60)
        print("          CAREDOSE - MEDICINE GUARDIAN CV          ")
        print("=" * 60)
        print(f"Alarm Interval   : Every {self.alarm_interval_seconds // 60} minutes ({self.alarm_interval_seconds} seconds)")
        print(f"Alarm Sound Mode : 60% Volume Gentle Chime Sound")
        print(f"Phone Locking    : ACTIVE until verified")
        print(f"Computer Vision  : OpenCV + Automatic Intake Action Confirmation")
        print("=" * 60)

        # Start background alarm timer
        self.alarm_system.start(callback=self.on_alarm_triggered)

        # Trigger first alarm cycle immediately if requested
        if trigger_immediately:
            print("\n[INIT] Triggering initial medicine alarm cycle for setup verification...")
            self.alarm_system.trigger_alarm()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Stopping Medicine Guardian App...")
            self.alarm_system.shutdown()
            print("Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CareDose Medicine Guardian - Python OpenCV Implementation")
    parser.add_argument("--interval", type=int, default=120, help="Alarm interval in seconds (default: 120 = 2 min)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index for OpenCV (default: 0)")
    parser.add_argument("--no-immediate", action="store_true", help="Do not trigger alarm on initial boot")
    
    args = parser.parse_args()

    app = MedicineGuardianApp(
        alarm_interval_seconds=args.interval,
        camera_index=args.cam
    )
    app.start(trigger_immediately=not args.no_immediate)
