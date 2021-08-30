import time
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
from scipy.spatial import distance as dist
import pyautogui as pag
from random import randint

def do(event_type, coordinate):
	if event_type == "click":
		do('hover', coordinate)
		pag.click(*coordinate)
	elif event_type == "hover":
		pag.moveTo(*coordinate)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ratio = (A + B) / (2.0 * C)
    return ratio

dlib.DLIB_USE_CUDA = True
shape_predictor = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
fileStream = False
frame_count = 0
prev_blink_frame = -1
curr_blink_frame = -1
while True:
    # pos = pag.position()
    # do('hover', (pos.x, pos.y))
    frame = 
    #frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1
    rects = detector(gray, 0)
    if len(rects):
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftratio = eye_aspect_ratio(leftEye)
        rightratio = eye_aspect_ratio(rightEye)
        avg_ratio = (leftratio + rightratio)/2
        if avg_ratio < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                curr_blink_frame = frame_count
                print(f"Blinked...{COUNTER}...{frame_count}")
                if (curr_blink_frame - prev_blink_frame) < 8 :
                    print('Double blink')
                    # do('click', (pos.x, pos.y))
                prev_blink_frame = curr_blink_frame
            COUNTER = 0
vs.stop()