import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=4)
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:  # If hands are detected
        for hand_landmark in results.multi_hand_landmarks:  # For each hand out of all detected hands
            
            for id,landmark in enumerate(hand_landmark.landmark):
                # print(id,landmark)
                h,w,c = img.shape #Dimensions of the image
                cx , cy = int(landmark.x*w), int(landmark.y*h) 
                print(id, cx, cy)
                # if id == 0:
                #     cv2.circle(img, (cx,cy), 5, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

    # displaying the fps
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_ITALIC, 1, (0,0,0), 2, )

    cv2.imshow("Hand-tracking with Python", img)
    cv2.waitKey(1)
