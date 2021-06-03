"""Converts image to image with just a face in byte format."""
import cv2
import numpy as np

# load classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def crop_to_face(image: np.ndarray, face: np.ndarray) -> np.ndarray:
    """
    Crops image to just the face and reshapes the image to 48 x 48.

    Parameters
    ----------
        image: np.ndarray
            An image in numpy array format

        face: np.ndarray
            Array that contains the point where te face starts and the width and height of the face.

    Return
    ------
        frame_of_face
            An image in numpy array format that is just the face
    """
    (x, y, width, height) = face
    frame_of_face = image[y: y + height, x: x + width]

    return frame_of_face

def reshape_image(image: np.ndarray) -> np.ndarray:
    """
    Reshapes the given image to a 48 x 48 pixels image

    Parameters
    ----------
        image: np.ndarray
            An image that needs to be reshaped to 48 x 48 pixels

    Return
    ------
        reshaped_frame
            The image reshaped to 48 x 48 pixels
    """
    reshaped_frame = cv2.resize(image, (48, 48))
    return reshaped_frame

def convert_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert the given image to bytes

    Parameters
    ----------
        image: np.ndarray

    Return
    ------
        byte_im

    """
    # convert np.array to one-dimensional numpy array
    _, im_buf_arr = cv2.imencode(".png", image)

    # convert one-dimensional numpy array to bytes
    byte_im = im_buf_arr.tobytes()
    return byte_im

def face_from_image(img_file: str) -> bytes:
    """
        Reads image, tries to detect faces, takes the face that is closest,
        then scales the image down to just the face and reshapes it as 48 x 48.
        Then it converts the image to bytes and returns the bytes.

        Parameters
        ----------
            img: str
                Path to image file

        Raise
        -----
            Error if no face can be detected

        Return
        ------
            byte_im
                An image of a face in a 48 x 48 image in byte format
    """

    # reads image
    image = cv2.imread(img_file)

    # make image gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get coordinates of detected face with a scaleFactor of 1.2 and a minNeighbors of 5
    detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # if no face is detected raise error
    if len(detected_faces) == 0:
        raise Exception("No face was detected in the image")

    closest_face = None
    closest_face = next((face for face in detected_faces if closest_face is not None if face[-1] > closest_face[-1] and face[-2] > closest_face[-2]), detected_faces[0])

    frame = crop_to_face(image, closest_face)
    reshaped_frame = reshape_image(frame)

    byte_im = convert_to_bytes(reshaped_frame)

    return byte_im
