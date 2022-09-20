import datetime
import tensorflow as tf
import tensorflow_addons as tfa

# Info about data
image_size = (416, 416)
input_shape = (*image_size, 3)
num_classes = 65



# default training config
default_training_config = { # lol thanks copilot lemme take a look at what this is
    'epochs': 10,
    'batch_size': 128*2,
    'monitor_metric': 'val_loss',
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'learning_rate': 1e-4,
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