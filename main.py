import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:  # If hands are detected
        for hand_landmark in results.multi_hand_landmarks:  # For each hand out of all detected hands
            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

    # displaying the fps
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.QT_FONT_LIGHT,
                1,  # Scale
                (3, 252, 15),5)

    cv2.imshow("Hand-tracking with Python", img)
    cv2.waitKey(1)
