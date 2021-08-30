import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from config_default import *
from torch.nn import functional as F
from torchvision import models

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.model = models.resnet18(pretrained=True)

        self.conv = nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # 25088 (512×7×7)
        return x

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()

        self.conv = ResNet()
        self.fc = nn.Sequential(
            # FC-F1
            # 25088
            nn.Dropout(0.4),
            nn.Linear(25088, 128),
            # 128
            nn.ReLU(inplace=True),

            # FC-F2
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            # 64
            nn.ReLU(inplace=True),
            # 64
        )
        

    def forward(self, x):
        # 3C x 224H x 224W
        x = self.conv(x)
        # 25088
        x = self.fc(x)
        # 64
        return x

class GridNet(nn.Module):
    def __init__(self):
        super(GridNet, self).__init__()
        self.conv = ResNet()
        self.fc = nn.Sequential(
            # FC-F1
            # 25088
            nn.Dropout(0.4),
            nn.Linear(25088, 256),
            # 256
            nn.ReLU(inplace=True),

            # FC-F2
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            # 128
            nn.ReLU(inplace=True),
            # 128
        )

    def forward(self, x):
        # 3C x 224H x 224W
        x = self.conv(x)
        # 25088
        x = self.fc(x)
        # 128
        return x

class InitialPredictionModel(nn.Module):
    def __init__(self):
        super(InitialPredictionModel, self).__init__()
        # 3Cx224Hx224W --> 25088
        self.eyeModel = ResNet()
        # 3Cx224Hx224W --> 64
        self.faceModel = FaceNet()
        # 3Cx224Hx224W --> 128
        self.gridModel = GridNet()


        # Joining both eyes
        self.eyesFC = nn.Sequential(
            # FC-E1
            nn.Dropout(0.4),
            # 50176
            nn.Linear(2 * 25088, 128),
            # 128
            nn.ReLU(inplace=True),
            # 128
        )

        # Joining everything
        self.fc = nn.Sequential(
            # FC1
            nn.Dropout(0.4),
            # 384 FC-E1 (128) + FC-F2(64) + FC-FG2(128)
            nn.Linear(128 + 64 + 128, 128),
            # 128
            nn.ReLU(inplace=True),
            # 128

            # FC2
            # 128
            nn.Dropout(0.4),
            nn.Linear(128, 2),
            # 2
        )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)  # CONV-E1 -> ... -> CONV-E4
        xEyeR = self.eyeModel(eyesRight)  # CONV-E1 -> ... -> CONV-E4

        # Cat Eyes and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)  # FC-E1

        # Face net
        xFace = self.faceModel(faces)  # CONV-F1 -> ... -> CONV-E4 -> FC-F1 -> FC-F2
        xGrid = self.gridModel(faceGrids)  # FC-FG1 -> FC-FG2

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)  # FC1 -> FC2

        return x
