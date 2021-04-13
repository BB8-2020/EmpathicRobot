"""First version of the face detection model using the OpenCV library."""

import cv2

# get laptop camera footage
cap = cv2.VideoCapture(0)

while True:
    # read and save footage
    _, img = cap.read()

    # show footage in new window
    cv2.imshow('Cam', img)

    # checks which key is pressed with a 30 milisecond delay
    k = cv2.waitKey(30)

    # breaks if key 27 (Esc key) is pressed
    if k == 27:
        break

# closes capturing
cap.release()