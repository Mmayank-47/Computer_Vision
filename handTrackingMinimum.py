import cv2
import mediapipe as mp
import time

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,9),(9,10),(10,11),(11,12), #middle
    (0,5),(5,6),(6,7),(7,8), #index
    (0,1),(1,2),(2,3),(3,4), #thumb
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# Landmarker options (VIDEO mode - synchronous)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Start camera
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp = int(time.time() * 1000)

        # Detect hands (synchronous)
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Draw landmarks if detected
        if result.hand_landmarks:
            h, w, _ = frame.shape

            for hand in result.hand_landmarks:
                points = []

                # Convert normalized → pixel
                for lm in hand:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw skeleton
                for connection in HAND_CONNECTIONS:
                    pt1 = points[connection[0]]
                    pt2 = points[connection[1]]
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        cv2.imshow("Hand Skeleton", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

cap.release()
cv2.destroyAllWindows()