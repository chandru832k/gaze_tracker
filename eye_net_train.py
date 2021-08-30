import torch
import torch.nn as nn

from config_default import DefaultConfig
from dataloader import *
from models.InitialPrediction import *
from utility_functions.load_model import *
from utility_functions.save_model import *

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0005
weight_decay = 0.001
EPOCHS = 10

model = InitialPredictionModel().to(device)

model.criterion = nn.MSELoss(reduction='mean').to(device)

model.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,)

def scale_xy_to_ab(value_xy, x, y, a, b):

    value_ab = ((value_xy - x) / (y - x)) * (b - a) + a

    return value_ab

def train():
    dataset = EVEProcessedDataset(
        './processedFrames/train01/step007_image_MIT-i2277207572')
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1)

    if config.eye_net_load_pretrained:
        load_weights_for_instance(model)
        print('Checkpoint loaded')
    for epoch in range(EPOCHS):
        print('Epoch ', epoch, 'going on')
        total_loss = 0
        print('Total length:', len(dataloader))
        print('Currently completed:', end=' ')
        for i, (face, left_eye, right_eye, face_grid, screen, pog) in enumerate(dataloader):
            print(i, end=' ')
            model.optimizer.zero_grad()

            initial_prediction = model(face, left_eye, right_eye, face_grid)
            
            screen_x, screen_y = pog[0].numpy()[0], pog[0].numpy()[1] 
            
            screen_x01, screen_y01 = scale_xy_to_ab(screen_x, 0, config.actual_screen_size[0], 0, 1), scale_xy_to_ab(screen_y, 0, config.actual_screen_size[1], 0, 1)
            
            pog_coordinates = np.array([screen_x01, screen_y01])
            pog_coordinates = torch.from_numpy(pog_coordinates)

            loss = model.criterion(initial_prediction.float(), pog.float())

            loss.backward()
            total_loss += loss.float()
            del loss

            model.optimizer.step()
            print('')
            print('     Prediction: ',initial_prediction)
            print('     Gaze point: ', pog)

        #saving the model
        save_weights_for_instance(model)
        print('Total Loss at end of ', epoch, 'is ', total_loss)


if __name__=='__main__':
    train()
