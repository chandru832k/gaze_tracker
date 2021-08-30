import torch
from config_default import DefaultConfig

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_weights_for_instance(model_instance):
    from models.eye_net import EyeNet
    from models.InitialPrediction import InitialPredictionModel
    from models.RefinedPrediction import RefineNet

    if isinstance(model_instance, InitialPredictionModel):
        model_fname = 'initial_prediction_eye_net.pt'
    elif isinstance(model_instance, EyeNet):
        model_fname = 'eve_eyenet_'
        model_fname += config.eye_net_rnn_type
        model_fname += '.pt'
    elif isinstance(model_instance, RefineNet):
        model_fname = 'eve_refinenet_'
        model_fname += config.refine_net_rnn_type
        model_fname += '_oa' if config.refine_net_do_offset_augmentation else ''
        model_fname += '_skip' if config.refine_net_use_skip_connections else ''
        model_fname += '.pt'
    else:
        raise ValueError('Cannot load weights for given model instance: %s' %
                         model_instance.__class__)

    model_path = config.checkpoint_path + model_fname

    # save the weights
    torch.save(model_instance.state_dict(), model_path)
