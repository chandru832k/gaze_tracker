import numpy as np
import torch
import torch.nn as nn
from config_default import *
from losses.angular import AngularLoss
from losses.cross_entropy import CrossEntropyLoss
from losses.euclidean import EuclideanLoss
from losses.l1 import L1Loss
from losses.mse import MSELoss
from utility_functions.load_model import load_weights_for_instance

from models.common import *
from models.eye_net import EyeNet
from models.RefinedPrediction import RefineNet

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cross_entropy_loss = CrossEntropyLoss()
euclidean_loss = EuclideanLoss()
angular_loss = AngularLoss()
mse_loss = MSELoss()
l1_loss = L1Loss()

mse_loss1 = nn.MSELoss()
cross_entropy_loss1 = torch.nn.CrossEntropyLoss()
class EVE(nn.Module):
    def __init__(self):
        super(EVE, self).__init__()

        # Initial prediction network
        self.eye_net = EyeNet()
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

    def forward(self, input_dict, output_dict, previous_output_dict=None):

        self.eye_net(input_dict, output_dict, side='left',
                     previous_output_dict=previous_output_dict)
        self.eye_net(input_dict, output_dict, side='right',
                     previous_output_dict=previous_output_dict)

        # for side in ('left', 'right'):
        #     output_dict[side + '_PoG_px_initial'] = to_screen_coordinates(
        #         output_dict[side + '_g_initial'])

        # print(output_dict['left_PoG_px_initial'])
        for side in ('left', 'right'):
            output_dict[side + '_g_initial'] = output_dict[side +'_g_initial'].clone().detach()

            scaled_x, scaled_y = output_dict[side + '_g_initial'][0].numpy(
            )[0], output_dict[side + '_g_initial'][0].numpy()[1]

            scaled_x, scaled_y = scale_xy_to_ab(
                scaled_x, -1, 1, 0, config.screen_size[0]), scale_xy_to_ab(scaled_y, -1, 1, 0, config.screen_size[1])

            output_dict[side + '_PoG_px_initial'] = torch.from_numpy(
                np.array([[scaled_x, scaled_y]]))

        output_dict['PoG_px_initial'] = torch.mean(torch.stack([
            output_dict['left_PoG_px_initial'],
            output_dict['right_PoG_px_initial'],
        ], axis=-1), axis=-1)

        
        output_dict['heatmap_initial'] = batch_make_heatmaps(
            output_dict['PoG_px_initial'], config.gaze_heatmap_sigma_initial) * input_dict['PoG_px_tobii_validity'].float().view(-1, 1, 1, 1)
        
        self.refine_net(input_dict, output_dict)

        # Step 3) Yield refined final PoG estimate(s)
        output_dict['PoG_px_final'] = soft_argmax(output_dict['heatmap_final'])

        if not config.skip_training:
            self.calculate_additional_labels(input_dict)

        # Calculate all loss terms and metrics (scores)
        self.calculate_losses_and_metrics(input_dict, output_dict)
        
        # Calculate the final combined (and weighted) loss
        full_loss = torch.zeros(()).to(device)
        output_dict['loss_terms'] = []

        # Add all losses for the GazeRefineNet
        # if 'loss_ce_heatmap_initial' in output_dict:
        #     full_loss += output_dict['loss_ce_heatmap_initial']
        #     output_dict['loss_terms'].append('loss_ce_heatmap_initial')

        # if 'loss_mse_heatmap_initial' in output_dict:
        #     full_loss += output_dict['loss_mse_heatmap_initial']
        #     output_dict['loss_terms'].append('loss_mse_heatmap_initial')

        if 'loss_ce_heatmap_final' in output_dict:
            full_loss += output_dict['loss_ce_heatmap_final']
            output_dict['loss_terms'].append('loss_ce_heatmap_final')

        if 'loss_mse_heatmap_final' in output_dict:
            full_loss += output_dict['loss_mse_heatmap_final']
            output_dict['loss_terms'].append('loss_mse_heatmap_final')

        if 'loss_mse_PoG_px_final' in output_dict:
            full_loss += output_dict['loss_mse_PoG_px_final']
            output_dict['loss_terms'].append('loss_mse_PoG_px_final')
        
        output_dict['full_loss'] = full_loss

        return output_dict

    def calculate_additional_labels(self, input_dict):
        input_dict['heatmap_initial'] = batch_make_heatmaps(input_dict['PoG_px_tobii'],
                                                            config.gaze_heatmap_sigma_initial)

        input_dict['heatmap_final'] = batch_make_heatmaps(input_dict['PoG_px_tobii'],
                                                          config.gaze_heatmap_sigma_initial)

        input_dict['heatmap_initial_validity'] = input_dict['PoG_px_tobii_validity']
        input_dict['heatmap_history_validity'] = input_dict['PoG_px_tobii_validity']
        input_dict['heatmap_final_validity'] = input_dict['PoG_px_tobii_validity']

    def calculate_losses_and_metrics(self, input_dict, output_dict):
        # Initial heatmap CE loss
        input_key = output_key = 'heatmap_initial'
        interm_key = 'heatmap_initial'
        if interm_key in output_dict and input_key in input_dict:
            output_dict['loss_ce_' + output_key] = cross_entropy_loss(
                output_dict[interm_key], input_key, input_dict,
            )
            output_dict['loss_mse_' + interm_key] = mse_loss(
                output_dict[interm_key], input_key, input_dict,
            )
        
        # Refined heatmap MSE loss
        input_key = interm_key = 'heatmap_final'
        if interm_key in output_dict and input_key in input_dict:
            output_dict['loss_ce_' + interm_key] = cross_entropy_loss(
                output_dict[interm_key], input_key, input_dict,
            )
            output_dict['loss_mse_' + interm_key] = mse_loss(
                output_dict[interm_key], input_key, input_dict,
            )

        # Initial gaze
        input_key = 'PoG_px_tobii'
        interm_key = 'PoG_px_initial'
        if interm_key in output_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                output_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                output_dict[interm_key], input_key, input_dict,
            )

        # Refine gaze
        input_key = 'PoG_px_tobii'
        interm_key = 'PoG_px_final'
        if interm_key in output_dict and input_key in input_dict:
            output_dict['loss_mse_' + interm_key] = mse_loss(
                output_dict[interm_key], input_key, input_dict,
            )
            output_dict['metric_euc_' + interm_key] = euclidean_loss(
                output_dict[interm_key], input_key, input_dict,
            )
