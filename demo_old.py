import time

import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import ImageGrab

from config_default import DefaultConfig
from dataloader import *
from models.EVEmodel import EVE
from utility_functions.face_utilities import *
from utility_functions.load_model import *
from utility_functions.save_model import *

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_res(cap, x,y):
    cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))

def demo():
    # Define model
    model = EVE()
    # print(model)
    model = model.to(device)
    # evaluate model:
    model.eval()
    
    config.skip_training = True
    imSize = (224, 224)
    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Specify video codec
    codec = cv2.VideoWriter_fourcc(*"XVID")

    # For plotting on screen
    x = []
    y = []

    # Duration for which to capture video
    
    capture_duration = 200
    start_time = time.time()
    while( int(time.time() - start_time) < capture_duration ):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        
        img = np.array(frame)
    
        shape_np, isValid = find_face_dlib(img)

        face_rect, left_eye_rect, right_eye_rect, isValid = rc_landmarksToRects(shape_np, isValid)

        # Crop images
        # imFace = cropImage(img, face_rect)
        imEyeL = cropImage(img, left_eye_rect)
        imEyeR = cropImage(img, right_eye_rect)

        imEyeL = cv2.resize(imEyeL, (256, 256), cv2.INTER_AREA)
        imEyeR = cv2.resize(imEyeR, (256, 256), cv2.INTER_AREA)
        
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
        input_dict['left_h'] = torch.from_numpy(np.array([[0, 0]]))
        input_dict['right_h'] = torch.from_numpy(np.array([[0, 0]]))
        input_dict['left_eye_patch'] = imEyeL.unsqueeze(0)
        input_dict['right_eye_patch'] = imEyeR.unsqueeze(0)
        input_dict['screen_frame'] = torch.from_numpy(resized_image).unsqueeze(0)
        input_dict['PoG_px_tobii_validity'] = torch.from_numpy(np.array([1]))
        
        with torch.no_grad():
            out_data = model(input_dict, output_dict)

        print('Coordinates:')
        print('X:', out_data['PoG_px_final'][0][0].float())
        print('Y:', out_data['PoG_px_final'][0][1].float())

        x.append(out_data['PoG_px_final'][0][0].item())
        y.append(out_data['PoG_px_final'][0][1].item())

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
    plt.plot(x, y, 'ro')
    plt.axis([0, config.actual_screen_size[0], 0, config.actual_screen_size[1]])
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis 
    plt.show()
    

if __name__=='__main__':
    demo()
    
