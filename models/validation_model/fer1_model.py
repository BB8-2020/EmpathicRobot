"""Simple implemention of a emotion recognizer with the face emotion recognizer library."""
# Bron: https://analyticsindiamag.com/face-emotion-recognizer-in-6-lines-of-code/

from fer import FER
import cv2


def main():
    """Run the validation model."""
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


if __name__ == "__main__":
    main()
