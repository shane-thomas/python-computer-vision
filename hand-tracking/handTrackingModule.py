import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.max = max_num_hands
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.max, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


def findHands(self, img, draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = self.hands.process(imgRGB)

    if results.multi_hand_landmarks:  # If hands are detected
        for hand_landmark in results.multi_hand_landmarks:  # For each hand out of all detected hands
            if draw:
                self.mpDraw.draw_landmarks(
                    img, hand_landmark, self.mpHands.HAND_CONNECTIONS)

            # for id, landmark in enumerate(hand_landmark.landmark):
            #     print(id,landmark)
            #     h, w, c = img.shape  # Dimensions of the image
            #     cx, cy = int(landmark.x*w), int(landmark.y*h)
            #     print(id, cx, cy)
            #     if id == 0:
            #       cv2.circle(img, (cx,cy), 5, (255, 0, 255), cv2.FILLED)
    return img

def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands()
        # displaying the fps
        currentTime = time.time()
        fps = 1 / (currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_ITALIC, 1, (0, 0, 0), 2, )

        cv2.imshow("Hand-tracking with Python", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()