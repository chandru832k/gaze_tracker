import os

import numpy as np
import torch
import torchvision
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset

from utility_functions.face_utilities import *


class EVEProcessedDataset(Dataset):
    # data path should be a stimulus folder containing preprocessed images of face, eyes, grid, screen and pog data
    def __init__(self, datapath):
        self.path = datapath
        self.cameras_to_use = 'webcam_c'
        self.imSize = (224, 224)
        self.color_space = 'RGB'
        self.split = 'train'

        pog_data = json_read(self.path + '/' + 'POG.json')
        
        self.pog = np.array([np.array([float(value) for value in point]) for point in pog_data['pog']])
        
        self.isValid = (np.array(pog_data['valid']) == "True").astype(int)
        
        self.number_of_frames = np.array(pog_data['number_of_frames'])

        face_path = self.path + '/' + 'face'
        left_eye_path = self.path + '/' + 'leftEye'
        right_eye_path = self.path + '/' + 'rightEye'
        face_grid_path = self.path + '/' + 'faceGrid'
        screen_path = self.path + '/' + 'screen.128x72'

        self.face = [None for i in range(self.number_of_frames)]
        self.left_eye = [None for i in range(self.number_of_frames)]
        self.right_eye = [None for i in range(self.number_of_frames)]
        self.face_grid = [None for i in range(self.number_of_frames)]

        self.screen = [None for i in range(self.number_of_frames)]

        self.normalize_image = normalize_image_transform(image_size=self.imSize, split=self.split, color_space=self.color_space)
        self.resize_transform = resize_image_transform(image_size=self.imSize)
            
        for i in range(self.number_of_frames):
            face_frame_path = face_path + '/' + str(i) + '.jpg'
            left_eye_frame_path = left_eye_path + '/' + str(i) + '.jpg'
            right_eye_frame_path = right_eye_path + '/' + str(i) + '.jpg'
            face_grid_frame_path = face_grid_path + '/' + str(i) + '.jpg'

            screen_frame_path = screen_path + '/' + str(i) + '.jpg'

            self.face[i] = self.loadImage(face_frame_path)
            self.left_eye[i] = self.loadImage(left_eye_frame_path)
            self.right_eye[i] = self.loadImage(right_eye_frame_path)
            self.face_grid[i] = self.loadImage(face_grid_frame_path)

            # Data Augmentation: Random Crop, Color Jitter
            # faceGrid mustn't have these augmentations
            self.face[i] = self.normalize_image(self.face[i])
            self.left_eye[i] = self.normalize_image(self.left_eye[i])
            self.right_eye[i] = self.normalize_image(self.right_eye[i])
            self.face_grid[i] = self.resize_transform(self.face_grid[i])

            self.screen[i] = np.array(self.loadImage(screen_frame_path))
            self.screen[i] = np.moveaxis(self.screen[i], -1, 0)

    def loadImage(self, path):
        try:
            im = PILImage.open(path).convert(self.color_space)
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
        return im

    def __getitem__(self, index):
        face = self.face[index]
        left_eye = self.left_eye[index]
        right_eye = self.right_eye[index]
        face_grid = self.face_grid[index]
        screen = torch.from_numpy(self.screen[index])

        pog = torch.from_numpy(self.pog[index])
        validity = torch.from_numpy(np.asarray([self.isValid[index]]))

        return face, left_eye, right_eye, face_grid, screen, pog, validity

    def __len__(self):
        return self.number_of_frames

if __name__=='__main__':
    dataset = EVEProcessedDataset('./processedFrames/train01/step007_image_MIT-i2277207572')
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=2)

    dataiter = iter(dataloader)
    data = dataiter.next()

    face, left_eye, right_eye, face_grid, screen, pog = data

    print(face)
