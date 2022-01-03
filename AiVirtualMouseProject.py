import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy



# variables
wCam, hCam = 640, 480
frameR = 100  # frame reduction
smoothening = 7

pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr , hScr)


while True:
    # 1. Find the hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        # index finger
        x1, y1 = lmList[8][1:]
        # middle finger
        x2, y2 = lmList[12][1:]

        # print(x1 ,y1 , x2, y2)
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only index finger: Moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            #5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            #6. Smoothing the values
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening
            #7. Move mouse
            autopy.mouse.move(wScr - cLocX, cLocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY

            # 8. Both fingers are up: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            # 9. find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # 10. Click mouse id distance is short
            if length < 37:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

                autopy.mouse.click()


        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
            # 9. find distance between fingers
            length, img, lineInfo = detector.findDistance(12, 16, img)

            # 10. Click mouse id distance is short
            if length < 37:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                right_button = autopy.mouse.Button.RIGHT
                autopy.mouse.click(right_button)

        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 1 and fingers[4] == 1:
            # 9. find distance between fingers
            length, img, lineInfo = detector.findDistance(16, 20, img)

            # 10. Click mouse id distance is short
            if length < 37:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                middle_button = autopy.mouse.Button.MIDDLE
                autopy.mouse.click(middle_button)


    # 11. Frame rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1) & 0xFF == ord('q')
