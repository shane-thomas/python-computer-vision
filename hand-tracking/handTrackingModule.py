import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,
                 mode=False,
                 max=2,
                 complexity=1,
                 detectionCon=0.5,
                 trackCon=0.5):
        self.mode = mode
        self.max = max
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max, self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:  # If hands are detected
            for hand_landmark in self.results.multi_hand_landmarks:  # For each hand out of all detected hands
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, hand_landmark, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        landmarks = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, landmark in enumerate(myHand.landmark):
                h, w, c = img.shape  # Dimensions of the image
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                # print(id, cx, cy)
                landmarks.append([id, cx, cy])
                # if id == 0:
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return landmarks


def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)
        if len(landmark_list)!=0:
            print(landmark_list[1])
            
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
