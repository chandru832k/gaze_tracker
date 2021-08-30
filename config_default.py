import glob
import json
import logging
import os
import sys
import zipfile


class DefaultConfig(object):
    # Data loading
    video_decoder_codec = 'libx264'  # libx264 | nvdec
    assumed_frame_rate = 10  # We will skip frames from source videos accordingly
    max_sequence_len = 30  # In frames assuming 10Hz
    face_size = [256, 256]  # width, height
    eyes_size = [128, 128]  # width, height
    screen_size = [128, 72]  # width, height
    actual_screen_size = [1920, 1080]  # DO NOT CHANGE
    camera_frame_type = 'eyes'  # full | face | eyes
    load_screen_content = True
    load_full_frame_for_visualization = False

    pixel_per_mm = [3.47197107, 3.47266881]

    train_cameras = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
    train_stimuli = ['image', 'video', 'wikipedia']
    test_cameras = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
    test_stimuli = ['image', 'video', 'wikipedia']

    # Inference
    input_path = ''
    output_path = ''

    # Training
    skip_training = True
    fully_reproducible = False  # enable with possible penalty of performance

    batch_size = 2
    weight_decay = 0.001
    num_epochs = 10.0

    train_data_workers = 1

    # Learning rate
    base_learning_rate = 0.0005

    @property
    def learning_rate(self):
        return self.batch_size * self.base_learning_rate
    # Available strategies:
    #     'exponential': step function with exponential decay
    #     'cyclic':      spiky down-up-downs (with exponential decay of peaks)
    num_warmup_epochs = 0.0  # No. of epochs to warmup LR from base to target
    lr_decay_strategy = 'none'
    lr_decay_factor = 0.5
    lr_decay_epoch_interval = 0.5

    # Gradient Clipping
    do_gradient_clipping = True
    gradient_clip_by = 'norm'  # 'norm' or 'value'
    gradient_clip_amount = 5.0

    # Eye gaze network configuration
    eye_net_load_pretrained = True
    eye_net_frozen = False
    eye_net_use_rnn = True
    eye_net_rnn_type = 'GRU'  # 'RNN' | 'LSTM' | 'GRU'
    eye_net_rnn_num_cells = 1
    eye_net_rnn_num_features = 128
    eye_net_static_num_features = 128
    eye_net_use_head_pose_input = True
    loss_coeff_PoG_cm_initial = 0.0
    loss_coeff_g_ang_initial = 1.0
    loss_coeff_pupil_size = 1.0

    # Conditional refine network configuration
    refine_net_enabled = True
    refine_net_load_pretrained = True

    refine_net_do_offset_augmentation = True
    refine_net_offset_augmentation_sigma = 3.0

    refine_net_use_skip_connections = True

    refine_net_use_rnn = True
    refine_net_rnn_type = 'CGRU'  # 'CRNN' | 'CLSTM' | 'CGRU'
    refine_net_rnn_num_cells = 1
    refine_net_num_features = 64
    loss_coeff_heatmap_ce_initial = 0.0
    loss_coeff_heatmap_ce_final = 1.0
    loss_coeff_heatmap_mse_final = 0.0
    loss_coeff_PoG_cm_final = 0.001

    # Heatmaps
    gaze_heatmap_size = [128, 72]
    gaze_heatmap_sigma_initial = 10.0  # in pixels
    gaze_heatmap_sigma_history = 3.0  # in pixels
    gaze_heatmap_sigma_final = 5.0  # in pixels
    gaze_history_map_decay_per_ms = 0.999

    # Evaluation
    test_num_samples = 128
    test_batch_size = 128
    test_data_workers = 0
    test_every_n_steps = 500
    full_test_batch_size = 128
    full_test_data_workers = 4

    codalab_eval_batch_size = 128
    codalab_eval_data_workers = 1

    # Checkpoints management
    checkpoints_save_every_n_steps = 100
    checkpoints_keep_n = 3
    resume_from = ''

    checkpoint_path = 'checkpoint/'
