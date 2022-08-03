import datetime

# Paths to things
dir_drive_home ='drive/MyDrive/MV1012_SBB_images/ML_projects'
dir_drive_data = f'{dir_drive_home}'
dir_google_cloud = 'gs://paleo-ml'
dir_data = f'{dir_google_cloud}/Processed_data/forams'
dir_dev_data = f'~/content/sample_data'
dir_training_logs = f'{dir_google_cloud}/Training_logs/'
dir_trained_models = f'{dir_drive_home}/Trained_models/'
dir_train_data = f'{dir_data}/train'
dir_val_data = f'{dir_data}/val'
dir_test_data = f'{dir_data}/test'
dir_debug_data = './training_data_debug'

# Info about data
image_size = (416, 416)
input_shape = (*image_size, 3)
num_classes = 65



# default training config
default_training_config = { # lol thanks copilot lemme take a look at what this is
    'epochs': 10,
    'batch_size': 128*2,

}
#     'learning_rate': 1e-4,
#     'steps_per_epoch': None,
#     'validation_steps': None,
#     'validation_freq': 1,
#     'callbacks': [
#         'tensorboard',
#         'checkpoint',
#         'early_stopping',
#         'reduce_lr',
#         'terminate_on_nan',
#     ],
#     'tensorboard': {
#         'log_dir': f'{dir_training_logs}/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
#     },
#     'checkpoint': {
#         'filepath': f'{dir_trained_models}/checkpoint-{datetime.now().strftime("%Y%m%d-%H%M%S")}.h5',
#         'save_best_only': True,
#         'save_weights_only': True,
#         'verbose': 1,
#     },
#     'early_stopping': {
#         'patience': 10,
#         'verbose': 1,
#     },
#     'reduce_lr': {
#         'factor': 0.1,
#         'patience': 5,
#         'min_lr': 1e-8,
#         'verbose': 1,
#     },
#     'terminate_on_nan': {
#         'verbose': 1,
#     },
# }