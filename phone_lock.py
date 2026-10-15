"""
Medicine Guardian - Phone Lock Controller Module
Enforces device lock until medicine intake is detected and verified by OpenCV.
Displays unlock pop-up window upon successful intake verification.
"""

import os
import sys
import ctypes
import threading
import time

# GUI message box support
try:
    import tkinter as tk
    from tkinter import messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


class PhoneLockManager:
    def __init__(self, enforce_system_lock: bool = False):
        """
        Initialize Phone Lock Manager.
        :param enforce_system_lock: If True on Windows, calls user32.LockWorkStation() when locked.
        """
        self.is_locked = False
        self.enforce_system_lock = enforce_system_lock
        self._lock_reason = ""
        self._lock_time = None

    def lock(self, reason: str = "Medicine intake pending"):
        """Lock the phone/device."""
        if self.is_locked:
            return

        self.is_locked = True
        self._lock_reason = reason
        self._lock_time = time.strftime("%H:%M:%S")

        print("\n" + "=" * 60)
        print("[DEVICE LOCKED] Status: LOCKED")
        print(f"   Reason: {reason}")
        print(f"   Locked at: {self._lock_time}")
        print("   ACTION REQUIRED: Move hand with medicine to mouth in camera view.")
        print("=" * 60 + "\n")

        # System-level lock on Windows if enabled
        if self.enforce_system_lock and sys.platform == "win32":
            try:
                print("[LOCK] Enforcing Windows System Workstation Lock...")
                ctypes.windll.user32.LockWorkStation()
            except Exception as e:
                print(f"[LOCK ERROR] System lock error: {e}")

    def unlock(self):
        """Unlock the phone/device upon verified intake and show unlock notification."""
        if not self.is_locked:
            return

        self.is_locked = False
        unlock_time = time.strftime('%H:%M:%S')

        print("\n" + "=" * 60)
        print("[PHONE IS UNLOCKED] Status: UNLOCKED")
        print("   Reason: OpenCV verified successful medicine intake!")
        print(f"   Device Access Restored at: {unlock_time}")
        print("=" * 60 + "\n")

        # Launch GUI popup in background thread so it displays immediately
        threading.Thread(target=self._show_unlock_dialog, args=(unlock_time,), daemon=True).start()

    def _show_unlock_dialog(self, unlock_time: str):
        """Display native desktop popup message: PHONE IS UNLOCKED!"""
        msg = f"🟢 Medicine intake verified successfully by Computer Vision!\n\nStatus: UNLOCKED\nAccess Restored at: {unlock_time}"
        title = "🔓 PHONE IS UNLOCKED!"

        if sys.platform == "win32":
            try:
                # Windows Native Message Box (MB_OK | MB_ICONINFORMATION)
                ctypes.windll.user32.MessageBoxW(0, msg, title, 0x40 | 0x10000) # MB_TOPMOST
                return
            except Exception:
                pass

        if HAS_TKINTER:
            try:
                root = tk.Tk()
                root.withdraw() # Hide root window
                root.attributes("-topmost", True)
                messagebox.showinfo(title, msg, parent=root)
                root.destroy()
            except Exception:
                pass

    def get_status_display(self) -> str:
        """Return formatted string for UI overlays."""
        if self.is_locked:
            return f"PHONE STATUS: 🔒 LOCKED ({self._lock_reason})"
        return "PHONE STATUS: 🔓 UNLOCKED (Medicine Taken)"


if __name__ == "__main__":
    locker = PhoneLockManager(enforce_system_lock=False)
    locker.lock("Test Alarm Triggered")
    time.sleep(1)
    locker.unlock()
