import torch
import torch.nn as nn

from config_default import DefaultConfig
from dataloader import *
from models.EVEmodel import EVE
from preprocess import *
from utility_functions.load_model import *
from utility_functions.save_model import *

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 1

# Define model
model = EVE()
# print(model)
model = model.to(device)

def test(person_id):

    directory = './dataset/'+person_id+'/'
    stimuli = [ name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) ]
    model.eval()

    for stimulus in stimuli:
        print("\n",stimulus)
        dataset = EVEProcessedDataset('./processedFrames/% s/% s'% (person_id, stimulus))
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1)
        
        print('Total length:', len(dataloader))
        print('Currently completed:', end=' ')

        ce_loss_value = 0
        mse_loss_value = 0
        mse_pog_px_loss_value = 0
        metric_euc_loss_value = 0

        for i, (face, left_eye, right_eye, face_grid, screen, pog, validity) in enumerate(dataloader):
            print(i, end=' ')
            # print(i)
            
            input_dict = {}
            output_dict = {}

            if face is None or left_eye is None or right_eye is None or face_grid is None or screen is None or pog is None or validity is None:
                continue

            input_dict['left_h'] = torch.from_numpy(np.array([[0, 0]]))
            input_dict['right_h'] = torch.from_numpy(np.array([[0, 0]]))
            input_dict['left_eye_patch'] = left_eye
            input_dict['right_eye_patch'] = right_eye
            input_dict['screen_frame'] = screen
            input_dict['PoG_px_tobii'] = pog
            input_dict['PoG_px_tobii_validity'] = validity

            with torch.no_grad():
                outputs = model(input_dict, output_dict)
            
            ce_loss_value += outputs['loss_ce_heatmap_final'].item()
            mse_loss_value += outputs['loss_mse_heatmap_final'].item()
            mse_pog_px_loss_value += outputs['loss_mse_PoG_px_final'].item()
            metric_euc_loss_value += outputs['metric_euc_PoG_px_final'].item()

            # print('')
            # print('     Prediction: ', output_dict['PoG_px_final'].float())
            # print('     Gaze point: ', pog)

        #saving the model
        save_weights_for_instance(model.eye_net)
        save_weights_for_instance(model.refine_net)
        print('\nTotal CE Loss for stimulus ', stimulus, 'is ', ce_loss_value/len(dataloader))
        print('Total MSE Loss for stimulus ', stimulus, 'is ', mse_loss_value/len(dataloader))
        print('Total MSE px loss for stimulus ', stimulus, 'is ', mse_pog_px_loss_value/(len(dataloader)*len(dataloader)))
        print('Total Euclidean px loss for stimulus ', stimulus, 'is ', metric_euc_loss_value/len(dataloader))

if __name__=='__main__':
    person_id = 'val01'
    camera_type = 'webcam_c'
    # preprocess(person_id, camera_type)
    test(person_id)
