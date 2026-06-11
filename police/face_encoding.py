import logging
from .face_recognition import get_encoding

def encode(key_points):
    """Encodes a list of 128-d numpy face descriptors into a custom string format."""
    if not key_points:
        return None  # no face detected
    encoded_string = ""
    for value in key_points[0]:
        svalue = str(value)
        if value < 0:
            svalue = svalue.replace('-', '1')  # Replace '-' with 1
        svalue = svalue.replace('.', '$')  # Replace . with $
        encoded_string = encoded_string + '@' + svalue
    return encoded_string

def get_key_points(image):
    """
    Passes the base64 form image to get facial key points.
    Returns encoded string or None if no face is detected.
    """
    result = get_encoding(image)
    if not result:
        return None  # no face found in image
    return encode(result)  # get key_points in string format

def decode(image):
    keypt = []
    keypt.append(image)
    encoded = []
    text = keypt[0].split('@')
    text = text[1:]
    for t in text:
        t = t.replace('$', '.')
        if t[0:1] == '1':
            t = '-' + t[1:]
        else:
            pass
        encoded.append(float(t))
    return encoded