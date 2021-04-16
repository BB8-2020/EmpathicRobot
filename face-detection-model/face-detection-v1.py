"""First version of the face detection model using the OpenCV library."""
import cv2

# load classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# get laptop camera footage
capture = cv2.VideoCapture(0)

while True:
    # read and save footage
    _, image = capture.read()

    # make image gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get coordinats of detected face with a scaleFactor of 2 and a minNeighbors of 4
    face = face_cascade.detectMultiScale(gray, 2, 4)

    # make one ractangle
    if len(face) > 0:
        (x, y, width, height) = face[0]
        cv2.rectangle(image, (x, y), (width + x, height + y), (255, 0, 255), 2)

    # show footage in new window
    cv2.imshow('image', image)

    # checks which key is pressed with a 30 milisecond delay
    k = cv2.waitKey(30)

    # breaks if key 27 (Esc key) is pressed
    if k == 27:
        break

# closes capturing
capture.release()