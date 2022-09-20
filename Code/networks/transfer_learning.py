#########
#TODO: FIX ALL BUGS RELATED TO PORTING AWAY FROM JUPYTER NOTEBOOK.
#TODO: REFACTOR TO STORE MODEL PARAMS IN SEPARATE FILE INSTEAD OF NAME
#TODO: CLEAN UP LOGIC FOR MODEL TRAINING
#TODO: MOVE MODEL TRAINING LOGIC TO SEPARATE FILE. THIS FILE SHOULD ONLY BE FOR TRANSFER LEARNING MODEL DEFINITIONS
#########


from keras_cv_attention_models import coatnet
# Import relevant transfer learning approaches
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.constants import *
from common.imports import *

# A string representing the name of the model to use
base_models = {
    'efficientnet_b0': EfficientNetB0,
    'efficientnetv2_b0': EfficientNetV2B0,
    'efficientnetv2_b1': EfficientNetV2B1,
    'efficientnetv2_b2': EfficientNetV2B2,
    'efficientnetv2_b3': EfficientNetV2B3,
    'efficientnetv2_s': EfficientNetV2S,
    'efficientnetv2_m': EfficientNetV2M,
    'efficientnetv2_l': EfficientNetV2L,
    'coatnet': coatnet,
}
def transfer_model_architecture(base_model_architecture, additional_layer_specs,**kwargs):
    """
    Create a sequential model on top of a headless base model architecture.

    Args:
        base_model_architecture (str): The name of the base model architecture to use.
        additional_layer_specs (list): A list of tuples, where each tuple is of the form (layer_type, layer_kwargs).
    """
    base_model = base_model_architecture(
        include_top = False,
        input_shape=input_shape
    )
    base_model.trainable = False
    model = keras.Sequential()
    model.add(base_model)
    for layer in additional_layer_specs:
        model.add(layer[0](**layer[1]))

    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dropout(0.2))
    # # model.add(keras.layers.Dense(512, activation='relu'))
    # # model.add(keras.layers.BatchNormalization())
    # # model.add(keras.layers.Dropout(0.3))
    # # model.add(keras.layers.Dense(256, activation='relu'))
    # # model.add(keras.layers.BatchNormalization())
    # # model.add(keras.layers.Dropout(0.3))
    # # model.add(keras.layers.Dense(128, activation='relu'))
    # # model.add(keras.layers.BatchNormalization())
    # # model.add(keras.layers.Dropout(0.3))
    # # model.add(keras.layers.Dense(64, activation='relu'))
    # # model.add(keras.layers.BatchNormalization())
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dense(num_classes, activation='softmax'))
    return model
