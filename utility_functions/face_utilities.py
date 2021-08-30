import json
import os
import sys

import cv2
import dlib
import numpy as np
import torchvision.transforms as transforms
from imutils import face_utils

file_dir_path = os.path.dirname(os.path.realpath(__file__))
landmarks_path = os.path.join(file_dir_path, '../metadata/shape_predictor_68_face_landmarks_GTX.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks_path)

def newFaceInfoDict(color="blue"):
    faceInfoDict = {
        "Face": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'Theta': [],
            'IsValid': []
        },
        "LeftEye": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'Theta': [],
            'IsValid': []
        },
        "RightEye": {
            'H': [],
            'W': [],
            'X': [],
            'Y': [],
            'Theta': [],
            'IsValid': []
        },
        "Color": color,
    }

    return faceInfoDict

def check_negative_coordinates(tup):
    isValid = True
    for idx in range(0, len(tup)):
        if tup[idx] < 0:
            isValid = False

    return isValid

def isCloserTo90(angle):
    if 90-angle < angle:
        return True
    return False

def getRect(data):
    # get the parameter of the small rectangle
    center, size, angle = data[0], data[1], data[2]
    # print(angle)
    # The function minAreaRect seems to give angles ranging in (-90, 0].
    # This is based on the long edge of the rectangle
    if isCloserTo90(angle):
        angle = -90 + angle
    
#     if angle > 0:
#     angle = -90 + angle
#     size = (size[1], size[0])

# this is working for left tilted faces
    return int(center[0]), int(center[1]), int(size[0]), int(size[1]), int(angle)

# def getRect(data):
#     # get the parameter of the small rectangle
#     center, size, angle = data[0], data[1], data[2]

#     # The function minAreaRect seems to give angles ranging in (-90, 0].
#     # This is based on the long edge of the rectangle
#     if angle < -45:
#         angle = 90 + angle
#         size = (size[1], size[0])

#     return int(center[0]), int(center[1]), int(size[0]), int(size[1]), int(angle)

def find_face_dlib(image):
    isValid = 0
    shape_np = None

    face_rectangles = detector(image, 0)

    if len(face_rectangles) == 1:
        isValid = 1
        rect = face_rectangles[0]

        shape = predictor(image, rect)
        shape_np = face_utils.shape_to_np(shape)

    return shape_np, isValid

def rc_landmarksToRects(shape_np, isValid):
    face_rect = (0, 0, 0, 0, 0)
    left_eye_rect = (0, 0, 0, 0, 0)
    right_eye_rect = (0, 0, 0, 0, 0)

    if isValid:
        (leftEyeLandmarksStart, leftEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rightEyeLandmarksStart, rightEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        left_eye_shape_np = shape_np[leftEyeLandmarksStart:leftEyeLandmarksEnd]
        right_eye_shape_np = shape_np[rightEyeLandmarksStart:rightEyeLandmarksEnd]

        face_rect = getRect(cv2.minAreaRect(shape_np))
        left_eye_rect = getRect(cv2.minAreaRect(left_eye_shape_np))
        right_eye_rect = getRect(cv2.minAreaRect(right_eye_shape_np))

        # ToDo enable negative coordinate check. Last value is theta which can be negative.
        isValid = check_negative_coordinates(face_rect[:-1]) and \
            check_negative_coordinates(left_eye_rect[:-1]) and \
            check_negative_coordinates(right_eye_rect[:-1])

    return face_rect, left_eye_rect, right_eye_rect, isValid

def rc_faceEyeRectsToFaceInfoDict(faceInfoDict, face_rect, left_eye_rect, right_eye_rect, isValid):
    face_dict = faceInfoDict["Face"]
    left_eye_dict = faceInfoDict["LeftEye"]
    right_eye_dict = faceInfoDict["RightEye"]

    face_dict['X'].append(face_rect[0])
    face_dict['Y'].append(face_rect[1])
    face_dict['W'].append(face_rect[2])
    face_dict['H'].append(face_rect[3])
    face_dict['Theta'].append(face_rect[4])
    face_dict['IsValid'].append(isValid)

    left_eye_dict['X'].append(left_eye_rect[0])
    left_eye_dict['Y'].append(left_eye_rect[1])
    left_eye_dict['W'].append(left_eye_rect[2])
    left_eye_dict['H'].append(left_eye_rect[3])
    left_eye_dict['Theta'].append(left_eye_rect[4])
    left_eye_dict['IsValid'].append(isValid)

    right_eye_dict['X'].append(right_eye_rect[0])
    right_eye_dict['Y'].append(right_eye_rect[1])
    right_eye_dict['W'].append(right_eye_rect[2])
    right_eye_dict['H'].append(right_eye_rect[3])
    right_eye_dict['Theta'].append(right_eye_rect[4])
    right_eye_dict['IsValid'].append(isValid)

    idx = len(face_dict['X']) - 1

    return faceInfoDict, idx

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = (rect[0], rect[1]), (rect[2], rect[3]), rect[4]

    center, size = tuple(map(int, center)), tuple(map(int, size))
    # get a square crop of the detected region with 10px padding
    size = (max(size) + 10, max(size) + 10)

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))
    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop

def getBox(face_rect):
    return ((face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), face_rect[4])

def generate_grid2(rect, webcam_image):
    im = np.zeros(webcam_image.shape, np.uint8)
    im[:] = (255,255,255)
    
    box = np.int0(cv2.boxPoints(getBox(rect)))
    im = cv2.drawContours(im, [box], 0, (0, 0, 0), -1)  # 2 for line, -1 for filled
    return im

def preparePath(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def logError(msg, critical=False):
    print(msg)
    if critical:
        sys.exit(1)
        
def json_read(filename):
    if not os.path.isfile(filename):
        logError('Warning: No such file %s!' % filename)
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        logError('Warning: Could not read file %s!' % filename)
        return None

    return data

def json_write(filename, data):
    with open(filename, "w") as write_file:
        json.dump(data, write_file)

def cropImage(img, bbox):
    bbox = np.array(bbox, int)
    if len(bbox) == 5:
        return crop_rect(img, bbox)
    else:
        aSrc = np.maximum(bbox[:2], 0)
        bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

        aDst = aSrc - bbox[:2]
        bDst = aDst + (bSrc - aSrc)

        res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
        res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0], :]

        return res

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def normalize_image_transform(image_size, split, color_space):
    normalize_image = []

    # Only for training
    if split == 'train':
        normalize_image.append(transforms.Resize(240))
        normalize_image.append(transforms.RandomCrop(image_size))

    # For training and Eval
    normalize_image.append(transforms.Resize(image_size))
    normalize_image.append(transforms.ToTensor())
    if color_space == 'RGB':
        normalize_image.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) # Well known ImageNet values

    return transforms.Compose(normalize_image)


def resize_image_transform(image_size):
    normalize_image = []
    normalize_image.append(transforms.Resize(image_size))
    normalize_image.append(transforms.ToTensor())
    return transforms.Compose(normalize_image)
