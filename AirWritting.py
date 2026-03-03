import cv2
import mediapipe as mp
import numpy as np
import cv2 as cv
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 1
)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
prev_x2, prev_y2 = 0,0

with HandLandmarker.create_from_options(options) as landmarker:
    color = (255,255,255)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv.flip(frame, 1)
        h,w,_ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_images = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        timestamp = int(time.time()*1000)

        result = landmarker.detect_for_video(mp_images,timestamp)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            index_tip = hand[8]
            index_base = hand[6]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            base_y = int(index_base.y * h)

            if y < base_y:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), color, 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            middle_tip = hand[12]
            middle_base = hand[10]

            ring_tip = hand[16]
            ring_base = hand[14]

            little_tip = hand[20]
            little_base = hand[18]

            thumb_tip = hand[4]
            thumb_base = hand[2]

            if middle_tip.y < middle_base.y and index_tip.y < index_base.y : #only middle and index are up
                color = (255,0,0)
            if middle_tip.y < middle_base.y and ring_base.y < ring_base.y and little_tip.y < little_base.y: #only middle and index are up
                color = (0,255,0)
            if thumb_tip.x > ring_base.x:
                color = (0,0,255)
            if middle_tip.y < middle_base.y and ring_base.y < ring_base.y and little_tip.y < little_base.y and index_tip.y < index_base.y:
                color = (255,255,255)







        frame =  cv2.add(frame, canvas)
        cv.imshow('Air Writing', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):
            canvas = np.zeros_like(canvas)
        elif key == ord('x'):
            break











