"""Converts image to image with just a face in byte format"""
import cv2

# load classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_face(img):
    """
        Reads image, tries to detect faces, takes the face that is closest,
        then scales the image down to just the face and reshapes it as 48 x 48.
        Then it converts the image to bytes and returns the bytes.
    """

    # reads image
    image = cv2.imread(img)

    # make image gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get coordinates of detected face with a scaleFactor of 1.2 and a minNeighbors of 5
    detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # if no face is detected
    if len(detected_faces) == 0:
        return

    closest_face = None
    # goes through all detected_faces and picks the face where the height and width is the highest
    for face in detected_faces:
        if closest_face is not None:
            if face[-1] > closest_face[-1] and face[-2] > closest_face[-2]:
                closest_face = face
        else:
            closest_face = face

    # for closest_face in detected_faces:
    (x, y, width, height) = closest_face
    frame = image[y: y+height, x: x+width]

    # reshapes image
    reshaped_frame = cv2.resize(frame, (48, 48))

    # convert np.array to one-dimensional numpy array
    _, im_buf_arr = cv2.imencode(".png", reshaped_frame)

    # convert one-dimensional numpy array to bytes
    byte_im = im_buf_arr.tobytes()

    return byte_im
