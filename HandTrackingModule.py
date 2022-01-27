import cv2
import time
import mediapipe as mp

# class is created to set all the parameters like detection confident etc
# and methods are defined like findHand , findPosition
class handDetector():
    def __init__(self, mode=False, maxHands=2,modelC=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        #here we have initiated mphand and mpdraw functions using mediapipe methods
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    #following block of code is responsible for tracking and marking the landmarks on the captured image
    def findHands(self, img, draw=True):
        #this imgRGb converts img to rgb format because mp works only on rgb formats
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLand in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLand, self.mpHands.HAND_CONNECTIONS)

        return img
    # the following function is used to find the landmarks on the hand here it is currently tracking tip of thumb
    def findPosition(self, img, handNo=0, draw=True):
        Position_List = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                Position_List.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return Position_List


def main():
    pTime = 0
    cTime = 0
    #cap is basic open cv function to capture image, and it is being set to 0 as webcam being used
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        Position_List = detector.findPosition(img)

        if len(Position_List) != 0:
            print(Position_List[4])
        #following helps to calculate fps i.e  1 divided by (currentTime -PreviousTime)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        #this prints out the fps on the screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)

#calling main function written above
if __name__ == "__main__":
    main()


