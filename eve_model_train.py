import torch
import torch.nn as nn

from config_default import DefaultConfig
from dataloader import *
from models.EVEmodel import EVE
from utility_functions.load_model import *
from utility_functions.save_model import *

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 10

# Define model
model = EVE()
# print(model)
model = model.to(device)

# Optimizer
model.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,)

model.criterion = nn.MSELoss(reduction='mean').to(device)

def scale_xy_to_ab(value_xy, x, y, a, b):

    value_ab = ((value_xy - x) / (y - x)) * (b - a) + a

    return value_ab

def train():
    dataset = EVEProcessedDataset(
        './processedFrames/train01/step007_image_MIT-i2277207572')
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1)

    for epoch in range(EPOCHS):
        print('Epoch ', epoch, 'going on')
        total_loss = 0
        print('Total length:', len(dataloader))
        print('Currently completed:', end=' ')
        mse_loss_value = 0
        
        for i, (face, left_eye, right_eye, face_grid, screen, pog, validity) in enumerate(dataloader):
            print(i, end=' ')
            # print(i)
            model.optimizer.zero_grad()

            input_dict = {}
            output_dict = {}

            input_dict['left_h'] = torch.from_numpy(np.array([[0, 0]]))
            input_dict['right_h'] = torch.from_numpy(np.array([[0, 0]]))
            input_dict['left_eye_patch'] = left_eye
            input_dict['right_eye_patch'] = right_eye
            input_dict['screen_frame'] = screen
            input_dict['PoG_px_tobii'] = pog
            input_dict['PoG_px_tobii_validity'] = validity
            outputs = model(input_dict, output_dict)
            
            print(screen.size())
            for loss_term in outputs['loss_terms']:
                outputs[loss_term].backward(retain_graph=True)
            
            model.optimizer.step()
            
            mse_loss_value += outputs['loss_mse_PoG_px_final']
            # print('')
            # print('     Prediction: ', output_dict['PoG_px_final'].float())
            # print('     Gaze point: ', pog)

        #saving the model
        save_weights_for_instance(model.eye_net)
        save_weights_for_instance(model.refine_net)
        print('Total Loss at end of ', epoch, 'is ', mse_loss_value/len(dataloader))

if __name__=='__main__':
    train()
