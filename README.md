# CareDose - 3-Stage Computer Vision Medicine Guardian (Python Implementation)

A dedicated, standalone Python implementation of the **Medicine Compliance & Phone Lock System** powered by OpenCV.

## 3-Step Verification Pipeline

The computer vision engine enforces **3 sequential checks**. If ANY check fails or is not detected, an **ERROR SIGN** is displayed, and the phone **REMAINS LOCKED**.

1. **Step 1: Hand Detection (`is_hand_detected`)**:
   - Detects the presence of the user's hand in the camera frame.
   - *Error if missing*: `❌ ERROR: HAND NOT DETECTED IN CAMERA VIEW`
2. **Step 2: Tablet in Hand Detection (`is_tablet_in_hand_detected`)**:
   - Takes time to scan and stabilize detection of a tablet/pill object held inside the hand contour.
   - *Error if missing/dropped*: `❌ ERROR: NO TABLET DETECTED IN HAND / DROPPED`
3. **Step 3: Swallow Confirmation (`is_swallow_confirmed`)**:
   - Verifies the hand and tablet reach the mouth target zone and that the tablet is swallowed (ingested into mouth) rather than dropped.
   - *Error if incomplete*: `❌ ERROR: BRING TABLET TO MOUTH & SWALLOW`

---

## Live HUD Display

The camera window renders a live 3-step checklist overlay:
```
[1] Hand Detected        : YES [OK] / NO [MISSING]
[2] Tablet in Hand       : YES [OK] / NO (Scanning: 80%)
[3] Swallow Confirmed    : YES [SWALLOWED] / NO [PENDING]
------------------------------------------------------
STATUS: ❌ ERROR: NO TABLET DETECTED IN HAND / DROPPED
```

Only when **ALL 3 CHECKS ARE GREEN (`YES [OK]`)** does the system confirm intake and **🔓 UNLOCK THE PHONE**.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r medicine_guardian_cv/requirements.txt
```

### 2. Run Main Application (Standard 6-Minute Alarm)
```bash
python medicine_guardian_cv/main.py
```

### 3. Test Mode (Quick 10-Second Alarm Interval)
```bash
python medicine_guardian_cv/main.py --interval 10
```
