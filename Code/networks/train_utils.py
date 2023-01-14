from inspect import signature
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.constants import *
from common.imports import *
from common import utils

def train_model(model, model_identifier, training_set, validation_set, dir_save, target='species', **kwargs):
    '''
    Train a model given a model and a string identifier for that model.
    Save the model, the tensorboard logs, and the model parameters in DIR_SAVE.

    Args:
    model (keras.Model): the model to train
    model_identifier (str): a string to identify the model
    dir_save(str): directory to save models, training logs, and whatnot to.

    Accepted kwargs:
    epochs (int): number of epochs to train (default in default_train_config['epochs'])
    monitor_metric (str): metric to monitor for early stopping and model checkpoint (default in default_train_config['monitor_metric'])
    early_stopping_patience (int): patience for early stopping (default 5)
    checkpoint(bool): whether to save a checkpoint (default True)

    Note: For TPU training, the tensorboard logs cannot be saved to a 

    return: history--the model's training history, for visualization
    '''
    supported_targets = ['species']
    if target not in supported_targets:
        raise ValueError(f"Target {target} not supported. Must be one of {supported_targets}")

    model_homedir = os.path.join(dir_save, model_identifier)
    dir_model = os.path.join(model_homedir, 'model')
    dir_tensorboard = os.path.join(model_homedir, datetime.now().strftime("%Y-%m-%d.%H-%M-%S"))
    dir_tensorboard_in_progress = os.path.join(model_homedir, datetime.now().strftime("%Y-%m-%d.%H-%M-%S") + '-in_progress')
    dir_params = os.path.join(model_homedir, 'params.json')
    
    checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    monitor_metric = 'val_accuracy'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        dir_model,
        save_best_only=True,
        monitor=monitor_metric,
        mode='max',
        options=checkpoint_options,
        save_freq='epoch',
    ) if kwargs.get('checkpoint', True) else None
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=kwargs.get('early_stopping_patience', 5), restore_best_weights=True, monitor=monitor_metric
    )

    get_supervised = lambda x: (x['image'], x[target])
    training_set = training_set.map(get_supervised).batch(default_training_config['batch_size'])
    validation_set = validation_set.map(get_supervised).batch(default_training_config['batch_size'])

    history = model.fit(
        training_set,
        epochs= kwargs.get('epochs', 10),
        verbose=1,
        callbacks=list(filter(lambda x: x is not None, 
            [
                # TqdmCallback(verbose=2),
                keras.callbacks.TensorBoard(log_dir=dir_tensorboard_in_progress) if kwargs.get('tensorboard', True) else None,
                checkpoint_cb,
                early_stopping_cb,
            ])),
        validation_data=validation_set,
    )

    bucket, dir_tensorboard = utils.parse_gcs_path(dir_tensorboard)
    bucket2, dir_tensorboard_in_progress = utils.parse_gcs_path(dir_tensorboard_in_progress)
    assert bucket == bucket2, f'Bucket mismatch: {bucket} and {bucket2}'
    utils.rename_cloud_dir(dir_tensorboard_in_progress, dir_tensorboard, bucket_name=bucket)
    return history

def compile_model(model, **kwargs):
    '''
    Compile a model with a given optimizer, loss, and metrics. 

    Kwargs:
    optimizer: tf.optim or string.
    loss: tf.loss or string.
    metrics: list of tf.metrics or list of strings.
    scheduler: string or LearningRateSchedule.
    '''
    kwargs['optimizer'] = kwargs.get('optimizer', default_training_config['optimizer'])
    kwargs['loss'] = kwargs.get('loss', default_training_config['loss'])
    kwargs['metrics'] = kwargs.get('metrics', default_training_config['metrics'])
    # TODO: add support for learning rate schedulers
    model.compile(
        **kwargs
    )
    return model

def load_train(model_identifier, save_new=False, **kwargs):
    """
    Load a model from the train_models directory and train it even more for the 
    given number of epochs. Saves the output either to the same model or to a 
    completely new one.
    """
    model = keras.models.load_model(f'drive/MyDrive/MV1012_SBB_images/ML_projects/Trained_models/{model_identifier}')
    if save_new:
        model_identifier = model_identifier + '-retrained' 
    if 'save' not in kwargs:
        kwargs['save'] = True
    history = train_model(model, model_identifier, **kwargs)
    return history


def compile_train(model_name, model=None, model_identifier=None, **kwargs):
    """
    DEPRECATED. Use compile_model and train_model instead.

    Compline and train the model specified by model_name. Returns the model 
    training history and also saves tensorboard logs and model callbacks based on model_identifier. 
    If model_identifier is not specified, model_name is used instead.
    
    Args:
    Model_name: a string to identify the model as it exists in complete_models, defined later
    model (opt): an actual model isntance in keras. Default None, which gets the model from complete models
    model_identifer: an override for the identifier of the model, which we will write to. 
    scheduler: string or LearningRateSchedule. If string, should be 'cosine learning rate'
    optimizer: tf.optim or string.

    All other kwargs passed to train_model. See docs there. 
    """
    model = complete_models[model_name] if model is None else model
    if model_identifier is None:
        model_identifier = model_name

    epochs = kwargs.get('epochs', 10)
    lr = kwargs.get('lr', kwargs.get('learning_rate', 1e-3))
    scheduler = kwargs.get('lr_scheduler', lr)
    if scheduler is None:
        scheduler = lr
    num_iter_steps = len(training_set) * epochs
    if isinstance(scheduler, str):
        scheduler_dict = {
            'cosine': keras.optimizers.schedules.CosineDecay(lr, num_iter_steps),
            'cosine-restart': keras.optimizers.schedules.CosineDecayRestarts(lr, len(training_set))
        }
        scheduler = scheduler_dict[scheduler]
    optimizer = kwargs.get('optimizer', kwargs.get('optim', 'adam'))
    if isinstance(optimizer, str):
        optim_dict = {
            'adam': tf.keras.optimizers.Adam,
            'adamw': tfa.optimizers.AdamW,
        }
        optim_args = {
            'adam': {},
            'adamw': {'weight_decay': 0.0}
        }
        optim_obj = optim_dict.get(optimizer)
        optim_arg = optim_args.get(optimizer)
    optimizer = optim_obj(
        learning_rate=scheduler,
        **optim_arg,
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
                'accuracy', 
                keras.metrics.TopKCategoricalAccuracy(
                    k=2, name="top2"
                ),
                keras.metrics.TopKCategoricalAccuracy(
                    k=3, name="top3"
                ),
                keras.metrics.precision(),
                keras.metrics.recall(),
                tfa.metrics.F1Score(
                    num_classes=2, average='macro', name='f1_score'
                ),
                ]
    )
    history = train_model(model, model_identifier, **kwargs)
    return history