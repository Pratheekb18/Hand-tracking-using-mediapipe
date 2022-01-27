import numpy as np
import cv2
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####################################
widthCam, heightCam = 640, 480
###################################

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)
# usage template code of pycaw
# pycaw is library used to control volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_Range = volume.GetVolumeRange()
vol = 0
volBar = 400
minVol = vol_Range[0]
maxVol = vol_Range[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    position_List = detector.findPosition(img, draw=False)
    if len(position_List) != 0:
        # print(position_List[4], position_List[8])  # this gets you the list of index 4 and index 8

        # here the x1 and y1 refer to x and y coordinate of index 4
        x1, y1 = position_List[4][1], position_List[4][2]

        # here the x2 and y2 refer to x and y coordinate of index 8
        x2, y2 = position_List[8][1], position_List[8][2]

        # to get center of distance between index_tip and thumb_tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # the following opencv function helps us draw circle around index_finger_tip[8] and thumb_tip[4]
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        # this is used to get the line between index [4] and [8]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # to mark down the center between two fingers
        cv2.circle(img, (cx, cy), 11, (255, 0, 255), cv2.FILLED)

        # to find the length of line between two fingers we use math library and use hypot function and pass coordinates
        # as values
        length = math.hypot(x2 - x1, y2 - y1)

        # hand range 50 to 300
        # volume range -65 to 0
        # we can convert hand range into volume range using numpy function

        vol = np.interp(length, [40, 225], [minVol, maxVol])
        volBar = np.interp(length, [40, 225], [400, 150])
        print(length, vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 11, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)
