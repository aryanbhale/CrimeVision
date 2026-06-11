import base64
import os
import pickle
import io
import cv2
from PIL import Image
import numpy as np
from .face_recognition import face_encodings


model_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'classifier.pkl')


def decode_base64(img):
    """Decode a base64 image (bytes or str repr) into a uint8 RGB numpy array for dlib."""
    if isinstance(img, bytes):
        raw = img
    else:
        img = img.strip()
        if img.startswith("b'") or img.startswith('b"'):
            img = img[2:-1]
        raw = base64.b64decode(img)
    arr = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
    return np.ascontiguousarray(arr, dtype=np.uint8)


def get_facial_points(img):
    return face_encodings(img)


def match(base64_image):
    if not os.path.isfile(model_name):
        return None  # classifier not trained yet
    with open(model_name, 'rb') as f:
        (le, clf) = pickle.load(f)

    image = decode_base64(str(base64_image))
    key_pts = get_facial_points(image)

    if not key_pts:
        return []  # no face detected in uploaded image

    matched = []
    closest_distances = clf.kneighbors(key_pts)
    is_recognized = [closest_distances[0][0][0] <= 0.5]
    if is_recognized[0]:
        predictions = [
            (le.inverse_transform([pred])) if rec else ("Unknown")
            for pred, rec in zip(clf.predict(key_pts), is_recognized)
        ]
        matched.append([predictions])
    return matched
