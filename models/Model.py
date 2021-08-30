import numpy as np
import torch
import torch.nn as nn
from config_default import *
from utility_functions.load_model import *

from models.common import *
from models.InitialPrediction import InitialPredictionModel
from models.RefinedPrediction import RefineNet

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()

        # Initial prediction network
        self.eye_net = InitialPredictionModel()
        if config.eye_net_load_pretrained:
            load_weights_for_instance(self.eye_net)
        if config.eye_net_frozen:
            for param in self.eye_net.parameters():
                param.requires_grad = False

        # Network to refine estimated gaze based on:
        #   a) history of point of gaze (PoG)
        #   b) screen content
        self.refine_net = RefineNet() if config.refine_net_enabled else None
        if config.refine_net_enabled and config.refine_net_load_pretrained:
            load_weights_for_instance(self.refine_net)

    def forward(self, input_dict, output_dict):

        initial_prediction_px_coordinates = self.eye_net(input_dict['face'], input_dict['left_eye'], input_dict['right_eye'], input_dict['face_grid'])

        initial_prediction_px_coordinates = initial_prediction_px_coordinates.detach()

        # scaled_x, scaled_y = initial_prediction_px_coordinates[0].numpy()[0], initial_prediction_px_coordinates[0].numpy()[1] 
            
        # scaled_x, scaled_y = scale_xy_to_ab(scaled_x, 0, config.screen_size[0], 0, 1), scale_xy_to_ab(scaled_y, 0, config.screen_size[1], 0, 1)

        output_dict['PoG_px_initial'] = initial_prediction_px_coordinates # to_screen_coordinates(initial_prediction_px_coordinates)

        output_dict['heatmap_initial'] = batch_make_heatmaps(output_dict['PoG_px_initial'], config.gaze_heatmap_sigma_initial)

        self.refine_net(input_dict, output_dict)

        # Step 3) Yield refined final PoG estimate(s)
        output_dict['PoG_px_final'] = soft_argmax(output_dict['heatmap_final'])

