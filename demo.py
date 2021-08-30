import time

import cv2
import dlib
import numpy as np
import pyautogui
import pyautogui as pag
import torch
import torch.nn as nn
from imutils.video import VideoStream
#from matplotlib import pyplot as plt
from PIL import ImageGrab
from scipy.spatial import distance as dist

from config_default import DefaultConfig
from dataloader import *
from models.EVEmodel import EVE
from utility_functions.face_utilities import *
from utility_functions.load_model import *
from utility_functions.save_model import *

#initializations
dlib.DLIB_USE_CUDA = True
shape_predictor = "metadata/shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
TOTAL = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

frame_count = 0
prev_blink_frame = -1
curr_blink_frame = -1

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def adjust_to_screen(x, y):
    # x += x*4
    # y += y*1.2
    return x, y

def set_res(cap, x,y):
    cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))

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

def demo():
    # Define model
    global frame_count, prev_blink_frame, curr_blink_frame, TOTAL, COUNTER
    model = EVE()
    # print(model)
    model = model.to(device)
    # evaluate model:
    model.eval()
    
    config.skip_training = True
    imSize = (224, 224)
    # define a video capture object
    # vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid = VideoStream(src=0).start()
    # Specify video codec
    codec = cv2.VideoWriter_fourcc(*"XVID")

    # For plotting on screen
    x = []
    y = []
    capture_duration = 50
    start_time = time.time()
    i = 0
    while( int(time.time() - start_time) < capture_duration ):
        # Capture the video frame
        # by frame
        frame = vid.read()
        i += 1
        # Display the resulting frame
        #cv2.imshow('frame', frame)
        
        img = np.array(frame)
    
        shape_np, isValid = find_face_dlib(img)

        if not isValid:
            print("No face found")
            continue

        face_rect, left_eye_rect, right_eye_rect, isValid = rc_landmarksToRects(shape_np, isValid)

        # Crop images
        # imFace = cropImage(img, face_rect)
        imEyeL = cropImage(img, left_eye_rect)
        imEyeR = cropImage(img, right_eye_rect)

        imEyeL = cv2.resize(imEyeL, (256, 256), cv2.INTER_AREA)
        imEyeR = cv2.resize(imEyeR, (256, 256), cv2.INTER_AREA)
        # cv2.imwrite(f"left_{i}.jpg", imEyeL)
        # cv2.imwrite(f"right_{i}.jpg", imEyeR)
        # i += 1

        # cv2.imwrite("capturedData/left/capturel" + str(i) + ".jpg", imEyeL)
        # cv2.imwrite("capturedData/right/capturer" + str(i) + ".jpg", imEyeL)
       
        imEyeL = PILImage.fromarray(imEyeL)
        imEyeR = PILImage.fromarray(imEyeR)

        imEyeL = normalize_image_transform(imSize, "test", "RGB")(imEyeL)
        imEyeR = normalize_image_transform(imSize, "test", "RGB")(imEyeR)

        # input_dict['PoG_px_tobii'] = pog
        # input_dict['PoG_px_tobii_validity'] = validity
            
        # Take screenshot using PyAutoGUI
        screen = pyautogui.screenshot()
        screen_frame = np.array(screen)
    
        # Convert it from BGR(Blue, Green, Red) to
        # RGB(Red, Green, Blue)
        screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2RGB)
        
        resized_image = image_resize(screen_frame, config.screen_size[0], config.screen_size[1])
        resized_image = np.moveaxis(resized_image, -1, 0)

        input_dict = {}
        output_dict = {}
        input_dict['left_h'] = torch.from_numpy(np.array([[0, 0]])).to(device)
        input_dict['right_h'] = torch.from_numpy(np.array([[0, 0]])).to(device)
        input_dict['left_eye_patch'] = imEyeL.unsqueeze(0).to(device)
        input_dict['right_eye_patch'] = imEyeR.unsqueeze(0).to(device)
        input_dict['screen_frame'] = torch.from_numpy(resized_image).unsqueeze(0).to(device)
        input_dict['PoG_px_tobii_validity'] = torch.from_numpy(np.array([1])).to(device)
        
        with torch.no_grad():
            out_data = model(input_dict, output_dict)

        # print('Coordinates:')
        # print('X:', out_data['PoG_px_final'][0][0].float())
        # print('Y:', out_data['PoG_px_final'][0][1].float())
        # print("Screen size :", config.actual_screen_size)
        x, y = adjust_to_screen(out_data['PoG_px_final'][0][0].float(), out_data['PoG_px_final'][0][1].float())
        # x.append(config.actual_screen_size[0] - out_data['PoG_px_final'][0][0].item())
        # y.append(config.actual_screen_size[1] - out_data['PoG_px_final'][0][1].item())
        print(x, y)
        # pag.moveTo(config.actual_screen_size[1]-y, config.actual_screen_size[0]-x)
        # print(config.actual_screen_size[0]-x, config.actual_screen_size[1]-y)
        
        #click part
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
                    # if (curr_blink_frame - prev_blink_frame) < 8 :
                        # print('Double blink')
                    pos = pag.position()
                    do('click', (pos.x, pos.y))
                    prev_blink_frame = curr_blink_frame
                COUNTER = 0

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.stop()
    # Destroy all the windows
    cv2.destroyAllWindows()

    # plt.plot(x, y, 'ro')
    # plt.axis([0, config.actual_screen_size[0], 0, config.actual_screen_size[1]])
    # plt.show()
    

if __name__=='__main__':
    demo()

