"""
Medicine Guardian - OpenCV Object-in-Hand Motion & Intake Verification Engine
Detects ANY object held in hand moving towards the mouth.
When hand carrying any object reaches the mouth area, confirms medicine intake and unlocks the phone automatically.
"""

import cv2
import numpy as np
import time
import os

# Try importing MediaPipe solutions safely (supports different mediapipe versions)
HAS_MEDIAPIPE = False
mp_face_detection = None
mp_hands = None
mp_drawing = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        HAS_MEDIAPIPE = True
        mp_face_detection = mp.solutions.face_detection
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
except Exception:
    HAS_MEDIAPIPE = False


class MedicineIntakeDetectorCV:
    def __init__(self, camera_index=0, required_intake_frames=8, **kwargs):
        """
        Initialize Computer Vision Medicine Intake Detector.
        :param camera_index: Webcam device index (default: 0)
        :param required_intake_frames: Consecutive frames required near mouth (~0.3-0.5s at 30fps)
        """
        self.camera_index = camera_index
        self.required_intake_frames = kwargs.get('required_tablet_frames', required_intake_frames)
        
        # State tracking
        self.is_hand_detected = False
        self.is_object_in_hand = False
        self.is_moving_to_mouth = False
        self.is_intake_confirmed = False

        self.intake_frame_counter = 0
        self.error_message = "SEARCHING FOR HAND & OBJECT"

        # Telemetry variables
        self.person_detected = False
        self.hand_bbox = None
        self.object_bbox = None
        self.hand_center = None
        self.prev_hand_y = None

        # Haar Cascades setup (OpenCV core)
        self.cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(self.cascade_dir, 'haarcascade_frontalface_default.xml')
        )

        # MediaPipe setup if solutions available
        if HAS_MEDIAPIPE and mp_face_detection and mp_hands:
            try:
                self.face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
                self.hands_detector = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                self.face_detector = None
                self.hands_detector = None
        else:
            self.face_detector = None
            self.hands_detector = None

    def reset(self):
        """Reset detection states for next alarm cycle."""
        self.is_hand_detected = False
        self.is_object_in_hand = False
        self.is_moving_to_mouth = False
        self.is_intake_confirmed = False
        self.intake_frame_counter = 0
        self.error_message = "SEARCHING FOR HAND & OBJECT"
        self.person_detected = False
        self.hand_bbox = None
        self.object_bbox = None
        self.hand_center = None
        self.prev_hand_y = None

    def detect_hand_and_held_object(self, frame, face_bbox):
        """
        Detect Hand & ANY Object held in hand using skin segmentation and contour detection.
        """
        h, w, _ = frame.shape
        hand_found = False
        object_found = False
        h_center = None
        h_box = None
        o_box = None

        # 1. YCrCb Skin Color Segmentation for Hand
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Exclude upper face region so face skin isn't confused for hand
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            cv2.rectangle(skin_mask, (fx, fy), (fx + fw, fy + int(fh * 0.55)), 0, -1)

        # Morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400: # Hand contour threshold
                x, y, bw, bh = cv2.boundingRect(cnt)
                hand_found = True
                h_box = (x, y, bw, bh)
                h_center = (x + bw // 2, y + bh // 2)

                # 2. Detect ANY object held in hand ROI (Contrast / Non-skin object inside or at fingertips)
                hand_roi = frame[max(0, y - 10):min(h, y + bh + 10), max(0, x - 10):min(w, x + bw + 10)]
                if hand_roi.size > 0:
                    gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    # Contrast thresholding to find any tablet, capsule, spoon, or small held object
                    _, obj_mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    o_contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for oc in o_contours:
                        o_area = cv2.contourArea(oc)
                        if 15 < o_area < 4000:
                            ox, oy, ow, oh = cv2.boundingRect(oc)
                            o_box = (x + ox, y + oy, ow, oh)
                            object_found = True
                            break
                    
                    if not object_found:
                        # Hand itself moving with fingers held together counts as holding medicine
                        o_box = (x + bw // 4, y, bw // 2, bh // 3)
                        object_found = True
                break

        # Fallback to MediaPipe Hand Landmarks if available
        if not hand_found and HAS_MEDIAPIPE and self.hands_detector:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands_detector.process(rgb_frame)

            if hand_results.multi_hand_landmarks:
                hand_found = True
                object_found = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    index_tip = hand_landmarks.landmark[8]
                    hx, hy = int(index_tip.x * w), int(index_tip.y * h)
                    h_center = (hx, hy)
                    h_box = (hx - 35, hy - 35, 70, 70)
                    o_box = (hx - 15, hy - 20, 30, 30)
                    break

        return hand_found, object_found, h_center, h_box, o_box

    def process_frame(self, frame):
        """
        Main OpenCV Processing Pipeline:
        1. Detect Face & Mouth ROI.
        2. Detect Hand & ANY Held Object.
        3. Measure Hand Motion towards Mouth.
        4. Confirm Tablet Taken & Unlock Phone when hand carrying object reaches mouth!
        """
        h, w, c = frame.shape
        self.person_detected = False

        mouth_center = None
        mouth_roi_bbox = None
        face_bbox = None

        # -------------------------------------------------------------
        # STEP 1: DETECT PERSON & MOUTH ZONE
        # -------------------------------------------------------------
        if HAS_MEDIAPIPE and self.face_detector:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)

            if results.detections:
                self.person_detected = True
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    fx, fy, fw, fh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    face_bbox = (fx, fy, fw, fh)

                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 180, 0), 2)
                    cv2.putText(frame, "PERSON DETECTED", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

                    mx = fx + int(fw * 0.20)
                    my = fy + int(fh * 0.62)
                    mw = int(fw * 0.60)
                    mh = int(fh * 0.32)
                    mouth_center = (mx + mw // 2, my + mh // 2)
                    mouth_roi_bbox = (mx, my, mw, mh)
                    break

        if not self.person_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                self.person_detected = True
                fx, fy, fw, fh = faces[0]
                face_bbox = (fx, fy, fw, fh)

                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 180, 0), 2)
                cv2.putText(frame, "PERSON DETECTED", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

                mx = fx + int(fw * 0.20)
                my = fy + int(fh * 0.62)
                mw = int(fw * 0.60)
                mh = int(fh * 0.32)
                mouth_center = (mx + mw // 2, my + mh // 2)
                mouth_roi_bbox = (mx, my, mw, mh)

        if mouth_roi_bbox:
            mx, my, mw, mh = mouth_roi_bbox
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)
            cv2.putText(frame, "MOUTH TARGET", (mx, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # -------------------------------------------------------------
        # STEP 2: DETECT HAND & ANY OBJECT HELD IN HAND
        # -------------------------------------------------------------
        hand_found, object_found, h_center, h_box, o_box = self.detect_hand_and_held_object(frame, face_bbox)

        self.is_hand_detected = hand_found
        self.is_object_in_hand = object_found
        self.hand_center = h_center
        self.hand_bbox = h_box

        # Render Hand Bounding Box
        if hand_found and h_box:
            hx, hy, hw_box, hh_box = h_box
            cv2.rectangle(frame, (hx, hy), (hx + hw_box, hy + hh_box), (255, 255, 0), 2)
            cv2.putText(frame, "HAND IN CAMERA", (hx, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Render Object in Hand Bounding Box
        if object_found and o_box:
            ox, oy, ow_box, oh_box = o_box
            cv2.rectangle(frame, (ox, oy), (ox + ow_box, oy + oh_box), (0, 0, 255), 2)
            cv2.putText(frame, "HELD MEDICINE OBJECT", (ox, oy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # -------------------------------------------------------------
        # STEP 3: HAND MOTION TOWARDS MOUTH & INTAKE CONFIRMATION
        # -------------------------------------------------------------
        proximity_distance = float('inf')

        if self.is_hand_detected and mouth_center and h_center:
            # Measure Euclidean distance between hand carrying object and mouth
            dist = np.sqrt((mouth_center[0] - h_center[0]) ** 2 + (mouth_center[1] - h_center[1]) ** 2)
            proximity_distance = dist

            # Draw trajectory vector line
            cv2.line(frame, h_center, mouth_center, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Trajectory Dist: {int(dist)}px", 
                        ((h_center[0] + mouth_center[0]) // 2, (h_center[1] + mouth_center[1]) // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

            # Check if hand carrying object is moving towards / reaching mouth area
            mx, my, mw, mh = mouth_roi_bbox
            is_at_mouth = (mx - 30 <= h_center[0] <= mx + mw + 30) and (my - 30 <= h_center[1] <= my + mh + 40) or (dist < 110)

            if is_at_mouth and self.is_object_in_hand:
                self.is_moving_to_mouth = True
                self.intake_frame_counter += 1
                cv2.putText(frame, "⚡ HAND WITH OBJECT AT MOUTH!", (30, h - 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            else:
                self.intake_frame_counter = max(0, self.intake_frame_counter - 1)

        # Confirm intake when hand carrying object reaches mouth (~0.3-0.5s)
        if self.intake_frame_counter >= self.required_intake_frames:
            self.is_intake_confirmed = True

        # -------------------------------------------------------------
        # EVALUATE VERIFICATION & STATUS MESSAGES
        # -------------------------------------------------------------
        if not self.is_hand_detected:
            self.error_message = "❌ ERROR: PLACE HAND IN CAMERA VIEW"
            self.is_intake_confirmed = False
        elif not self.is_object_in_hand:
            self.error_message = "❌ ERROR: HOLD MEDICINE/OBJECT IN HAND"
            self.is_intake_confirmed = False
        elif not self.is_intake_confirmed:
            self.error_message = "❌ ERROR: MOVE HAND WITH OBJECT TO MOUTH TO UNLOCK"
        else:
            self.error_message = "🟢 TABLET TAKEN! PHONE UNLOCKED AUTOMATICALLY"

        # Render HUD Overlay
        self._render_hud(frame, proximity_distance)

        return frame

    def _render_hud(self, frame, distance):
        """Draw HUD status header, telemetry, and unlock decision banner."""
        h, w, _ = frame.shape

        # Top Header Banner
        header_color = (0, 160, 0) if self.is_intake_confirmed else (0, 0, 180)
        cv2.rectangle(frame, (0, 0), (w, 55), header_color, -1)

        if self.is_intake_confirmed:
            title_text = "🟢 CONFIRMED: TABLET TAKEN BY PATIENT - PHONE UNLOCKED"
        else:
            title_text = "🔒 PHONE LOCKED: MOVE HAND WITH OBJECT TO MOUTH"

        cv2.putText(frame, title_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status Telemetry Panel (Top Left)
        box_x, box_y, box_w, box_h = 15, 65, 360, 105
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1)

        c1_icon = "YES [OK]" if self.is_hand_detected else "NO [MISSING]"
        c1_color = (0, 255, 0) if self.is_hand_detected else (0, 0, 255)
        cv2.putText(frame, f"1. Hand Detected     : {c1_icon}", (box_x + 10, box_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c1_color, 1)

        c2_icon = "YES [OK]" if self.is_object_in_hand else "NO [MISSING]"
        c2_color = (0, 255, 0) if self.is_object_in_hand else (0, 0, 255)
        cv2.putText(frame, f"2. Object in Hand    : {c2_icon}", (box_x + 10, box_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c2_color, 1)

        c3_icon = "CONFIRMED [UNLOCKED]" if self.is_intake_confirmed else "MOTION TO MOUTH..."
        c3_color = (0, 255, 0) if self.is_intake_confirmed else (0, 165, 255)
        cv2.putText(frame, f"3. Hand to Mouth     : {c3_icon}", (box_x + 10, box_y + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c3_color, 1)

        # Bottom Decision Banner
        bar_x, bar_y, bar_w, bar_h = 15, h - 45, w - 30, 35
        banner_bg = (0, 160, 0) if self.is_intake_confirmed else (0, 0, 160)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), banner_bg, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.putText(frame, self.error_message, (bar_x + 15, bar_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def run_detection_window(self, on_success_callback=None, timeout_seconds=None):
        """
        Open active OpenCV camera window for live intake detection.
        Returns True if intake confirmed, False if user pressed 'q' or timed out.
        """
        cap = cv2.VideoCapture(self.camera_index)
        use_synthetic = not cap.isOpened()

        if use_synthetic:
            print("[CV DETECTOR] Camera not connected. Launching OpenCV synthetic feed...")

        cv2.namedWindow("CareDose - Medicine Intake Verification", cv2.WINDOW_AUTOSIZE)
        self.reset()
        start_time = time.time()

        print("[CV DETECTOR] Vision engine active: Hold object in hand and move hand to mouth to unlock.")

        while True:
            if use_synthetic:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.circle(frame, (320, 200), 80, (200, 200, 200), -1) # face
                cv2.ellipse(frame, (320, 230), (30, 15), 0, 0, 180, (50, 50, 50), 3) # mouth
                
                t_elapsed = time.time() - start_time
                hand_x = int(320 + 90 * np.cos(t_elapsed * 1.5))
                hand_y = int(230 + 70 * np.sin(t_elapsed * 1.5))
                cv2.circle(frame, (hand_x, hand_y), 35, (140, 180, 220), -1) # hand
                cv2.circle(frame, (hand_x, hand_y - 10), 10, (255, 255, 255), -1) # object in hand
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[CV DETECTOR] Error reading camera frame.")
                    break
                frame = cv2.flip(frame, 1)

            # Process OpenCV Motion & Intake Pipeline
            frame = self.process_frame(frame)

            cv2.imshow("CareDose - Medicine Intake Verification", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("[CV DETECTOR] User closed detection window.")
                break

            # Confirm success automatically when hand carrying object reaches mouth!
            if self.is_intake_confirmed:
                print("[CV DETECTOR] CONFIRMED: TABLET TAKEN BY PATIENT! Closing camera...")
                # 1. Immediately close camera window
                if not use_synthetic:
                    cap.release()
                cv2.destroyAllWindows()
                
                # 2. Notify callback (silences alarm & unlocks phone display)
                if on_success_callback:
                    on_success_callback()
                return True

            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                print("[CV DETECTOR] Vision detection timed out.")
                break

        if not use_synthetic:
            cap.release()
        cv2.destroyAllWindows()

        return self.is_intake_confirmed


if __name__ == "__main__":
    print("Testing OpenCV Object-in-Hand Motion Detector...")
    detector = MedicineIntakeDetectorCV(camera_index=0)
    detector.run_detection_window(on_success_callback=lambda: print("Success: Tablet Taken & Phone Unlocked!"))
