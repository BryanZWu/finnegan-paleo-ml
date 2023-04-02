import tensorflow as tf

# Info about data
image_size = (224, 224)
input_shape = (*image_size, 3)
SPECIES_CLASSES = ['NOT FORAM', 'suggrunda eckisi', 'bulimina exilis',
  'nonionella stella', 'melonis affinis', 'bolivina argentea', 'fursenkoina bradyi',
  'bolivina seminuda', 'chilostomella oolina', 'bolivina sp. a', 'bolivina seminuda var. humilis',
  'bolivina spissa', 'cibicidoides sp. a', 'bolivina alata', 'cibicidoides wuellerstorfi',
  'chilostomella ovoidea', 'bolivina pacifica', 'nonionella decora', 'cassidulina crassa',
  'globocassidulina subglobosa', 'cassidulina minuta', 'epistominella exigua', 'oolina squamosa',
  'pyrgo murrhina', 'pullenia elegans', 'buccella peruviana', 'gyroidina subtenera',
  'bolivinita minuta', 'cassidulina carinata', 'alabaminella weddellensis', 'anomalinoides minimus',
  'uvigerina peregrina', 'pullenia bulloides', 'lenticulina sp. a', 'epistominella pulchella',
  'uvigerina interruptacostata', 'cassidulina auka', 'fursenkoina complanata', 'epistominella sp. a',
  'melonis pompilioides', 'laevidentalina sp. a', 'bolivina interjuncta', 'praeglobobulimina spinescens',
  'cassidulina delicata', 'globocassidulina neomargareta', 'triloculina trihedra', 'globobulimina barbata',
  'bolivina ordinaria', 'astrononion stellatum', 'epistominella obesa', 'epistominella pacifica',
  'fursenkoina pauciloculata', 'pyrgo sp. a', 'epistominella sandiegoensis', 'angulogerina angulosa']

num_classes = len(SPECIES_CLASSES)
# Training

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