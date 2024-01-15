import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
     
    cv2.imshow("Hand-tracking with Python", img)
    cv2.waitKey(1)