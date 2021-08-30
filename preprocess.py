import json
import os
import re
import shutil
import sys

import cv2
import h5py
import numpy as np
from PIL import ExifTags
from PIL import Image as PILImage

from config_default import DefaultConfig
from utility_functions.face_utilities import *

config = DefaultConfig()

source_to_fps = {
    'screen': 30,
    'basler': 60,
    'webcam_l': 30,
    'webcam_c': 30,
    'webcam_r': 30,
}

source_to_interval_ms = dict([
    (source, 1e3 / fps) for source, fps in source_to_fps.items()
])

max_sequence_len = 30
assumed_frame_rate = 10

def get_frames_camera(input_path, output_path):
    vidcap = cv2.VideoCapture(input_path+'.mp4')
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    # print('fps = ' + str(fps))
    # print('number of frames = ' + str(frame_count))
    # print('duration (S) = ' + str(duration))
    
    success,image = vidcap.read()
    count = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    while success:
        if image.shape == (1080, 1920, 3) or image.shape == (1920, 1080, 3):
            # converting image to 720p as most web cams are 720p
            image = image_resize(image, 1280, 720)
        cv2.imwrite(output_path+ '/' + str(count)+".jpg", image)     # save frame as JPEG file
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

def get_frames_screen(input_path, output_path):
    vidcap = cv2.VideoCapture(input_path+'.mp4')
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    # print('fps = ' + str(fps))
    # print('number of frames = ' + str(frame_count))
    # print('duration (S) = ' + str(duration))
    
    success,image = vidcap.read()
    count = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    while success:
        cv2.imwrite(output_path+ '/' + str(count)+".jpg", image)     # save frame as JPEG file
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

def save_pog(input_path, output_path):
    def traverse_datasets(hdf_file):
        def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                path = f'{prefix}/{key}'
                if isinstance(item, h5py.Dataset): # test for dataset
                    yield (path, item)
                elif isinstance(item, h5py.Group): # test for group (go down)
                    yield from h5py_dataset_iterator(item, path)

        for path, _ in h5py_dataset_iterator(hdf_file):
            yield path

    with h5py.File(input_path, 'r') as f:
        # Get point of gaze coordinates
        pog_dict = dict()
        pog_dict['pog'] = []
        pog_dict['valid'] = []
        
        data = f.get('face_PoG_tobii').get('data')
        pog = np.array(data)

        pog_dict['number_of_frames'] = pog.shape[0]

        for index, point in enumerate(pog):
            pog_dict['pog'].append((str(point[0]), str(point[1])))

        data = f.get('face_PoG_tobii').get('validity')
        validity = np.array(data)
        
        for index, isValid in enumerate(validity):
            pog_dict['valid'].append((str(isValid)))
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        out_file = open(output_path+'/'+'POG.json', "w") 
        json.dump(pog_dict, out_file) 
        out_file.close()

def ROIExtractionTask(input_path, output_path):

    pog_file = open(input_path+'POG.json')
    camera_type = 'webcam_c'
    data = json.load(pog_file)
    
    number_of_frames = int(data['number_of_frames'])
    dlibDir = input_path

    # prepape output paths
    facePath = preparePath(output_path + '/' + 'face')
    leftEyePath = preparePath(output_path + '/' + 'leftEye')
    rightEyePath = preparePath(output_path + '/' + 'rightEye')
    
    for i in range(number_of_frames):
        image_path = input_path+'/'+ camera_type + '/' + str(i) + '.jpg'

        if not os.path.isfile(image_path):
            logError('Warning: Could not read image file %s!' % image_path)
            continue
        image = PILImage.open(image_path)
        
        if image is None:
            logError('Warning: Could not read image file %s!' % image_path)
            continue

        img = np.array(image.convert('RGB'))

        shape_np, isValid = find_face_dlib(img)

        face_rect, left_eye_rect, right_eye_rect, isValid = rc_landmarksToRects(shape_np, isValid)

        if not isValid:
            continue

        # Crop images
        imFace = cropImage(img, face_rect)
        imEyeL = cropImage(img, left_eye_rect)
        imEyeR = cropImage(img, right_eye_rect)

        # Rotation Correction FaceGrid
        faceGridPath = preparePath(output_path + '/' + 'faceGrid')
        imFaceGrid = generate_grid2(face_rect, img)

        imFace = cv2.resize(imFace, (256, 256), cv2.INTER_AREA)
        imEyeL = cv2.resize(imEyeL, (256, 256), cv2.INTER_AREA)
        imEyeR = cv2.resize(imEyeR, (256, 256), cv2.INTER_AREA)
        imFaceGrid = cv2.resize(imFaceGrid, (256, 256), cv2.INTER_AREA)

        # Save images
        PILImage.fromarray(imFace).save(facePath + '/' + '%d.jpg' % i, quality=95)
        PILImage.fromarray(imEyeL).save(leftEyePath + '/' + '%d.jpg' % i, quality=95)
        PILImage.fromarray(imEyeR).save(rightEyePath + '/' + '%d.jpg' % i, quality=95)
        PILImage.fromarray(imFaceGrid).save(faceGridPath + '/' + '%d.jpg' % i, quality=95)

def preprocess(person_id, camera_type):
    directory = './dataset/'+person_id+'/'
    stimuli = [ name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) ]
    # print(stimuli)

    for stimulus in stimuli:    
        input_path = './dataset/'+person_id+'/'+stimulus
        output_path = './processedFrames/'+person_id+'/'+stimulus

        # save point of gaze in json file
        save_pog(input_path + '/' + camera_type+'.h5', output_path)

        # extract frames from video
        get_frames_camera(input_path + '/' + camera_type, output_path + '/' + camera_type)
        get_frames_screen(input_path + '/' + 'screen.128x72', output_path + '/' + 'screen.128x72')

        # region of interest extraction
        ROIExtractionTask(output_path + '/', output_path)
        return
