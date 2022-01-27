import cv2
import mediapipe as mp
import time #to check framerate

#to capture the image we use webcam
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands =mpHands.Hands(False)
#if kept  in static mode whole time it will be detecting ,
# but if kept false it detects only if confidence is high
mpDraw = mp.solutions.drawing_utils #this lets you add mapping drawing on the live image obtained

pTime =0
cTime =0

while True:
    success,img = cap.read()
    #send in rgb image
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #CONVERT IMAGE INTO RGB BECAUSE ONLY RGB INPUTIS TAKEN
    results = hands.process(imgRGB)  # we are calling hands object

    #for loop to check if there is multiple hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):

                h ,w ,c =img.shape   #takes in shape of image input using shape method
                cx,cy =int(lm.x*w),int(lm.y*h) #gives value of cx and cy
                print(id,cx,cy)

                cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) #draws landmark points and all the connections


    #to put out fps

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255, 0, 0),3)



    cv2.imshow("image",img)
    cv2.waitKey(1)
#the above block of code is to access the webcam
