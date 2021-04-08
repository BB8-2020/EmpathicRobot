"""Simple implemention of a emotion recognizer with the face emotion recognizer library."""

from fer import FER
import cv2

# calls camera on device
cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

# shows the view of the camera
while True:
    _, img = cap.read()
    cv2.imshow('ik', img)
    k = cv2.waitKey(30)
    if k == 27:
        break
    # print detected emotion
    print(detector.detect_emotions(img))

""" Bron: https://analyticsindiamag.com/face-emotion-recognizer-in-6-lines-of-code/"""
