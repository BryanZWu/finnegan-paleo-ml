import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.constants import *
from common.imports import *
from common import utils

def train_model(model, model_identifier, training_set, validation_set, dir_save, target='species', **kwargs):
    '''
    Train a model given a model and a string identifier for that model.
    Save the model, the tensorboard logs, and the model parameters in DIR_SAVE.

    When training with multiple targets, be sure to adapt the loss function to accept the 
    concatenated targets in the order of the targets in the list.

    Args:
    model (keras.Model): the model to train
    model_identifier (str): a string to identify the model
    dir_save(str): directory to save models, training logs, and whatnot to.

    Accepted kwargs:
    epochs (int): number of epochs to train (default in default_train_config['epochs'])
    monitor_metric (str): metric to monitor for early stopping and model checkpoint (default in default_train_config['monitor_metric'])
    print_metrics (list): metrics to print during training (default in default_train_config['print_metrics'])
    early_stopping_patience (int): patience for early stopping (default 5)
    checkpoint(bool): whether to save a checkpoint (default True)
    target (str): target to train on. Can be a string or a list of strings. (default 'species')

    Note: For TPU training, the tensorboard logs cannot be saved to a local directory.

    return: history--the model's training history, for visualization
    '''
    supported_targets = ['species', 'chamber_broken']
    if isinstance(target, list):
        for t in target:
            if t not in supported_targets:
                raise ValueError(f"Target {t} not supported. Must be one of {supported_targets}")
    else:
        if target not in supported_targets:
            raise ValueError(f"Target {target} not supported. Must be one of {supported_targets}")

    model_homedir = os.path.join(dir_save, model_identifier)
    dir_model = os.path.join(model_homedir, 'model')
    dir_tensorboard = os.path.join(model_homedir, datetime.now().strftime("%Y-%m-%d.%H-%M-%S"))
    dir_tensorboard_in_progress = os.path.join(model_homedir, datetime.now().strftime("%Y-%m-%d.%H-%M-%S") + '-in_progress')
    dir_params = os.path.join(model_homedir, 'params.json')
    
    # Callbacks for checkpointing, early stopping, and reducing learning rate
    checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    monitor_metric = kwargs.get('monitor_metric', 'val_accuracy')
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
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric, factor=kwargs['reduce_lr_factor'], patience=2, min_lr=1e-6
    ) if kwargs.get('reduce_lr_factor', None) is not None else None

    get_supervised_single_target = lambda x: (x['image'], x[target])
    # Multi target is jsut the concatenated tensors. Since they are ordinal, we concat two scalars into a tensor.
    get_supervised_multi_target = lambda x: (x['image'], tf.convert_to_tensor([x[t] for t in target]))
    get_supervised = get_supervised_single_target if isinstance(target, str) else get_supervised_multi_target

    training_set = training_set.map(get_supervised).batch(default_training_config['batch_size'])
    validation_set = validation_set.map(get_supervised).batch(default_training_config['batch_size'])

    history = model.fit(
        training_set,
        epochs= kwargs.get('epochs', 10),
        verbose=1,
        callbacks=list(filter(lambda x: x is not None, 
            [
        #         # TqdmCallback(verbose=2),
                keras.callbacks.TensorBoard(log_dir=dir_tensorboard_in_progress) if kwargs.get('tensorboard', True) else None,
                checkpoint_cb,
                early_stopping_cb,
                reduce_lr_cb,
            ])),
        validation_data=validation_set,
    )

    bucket, dir_tensorboard = utils.parse_gcs_path(dir_tensorboard)
    bucket2, dir_tensorboard_in_progress = utils.parse_gcs_path(dir_tensorboard_in_progress)
    assert bucket == bucket2, f'Bucket mismatch: {bucket} and {bucket2}'
    utils.rename_cloud_dir(dir_tensorboard_in_progress, dir_tensorboard, bucket_name=bucket)

    # Save the model hyperparameters. Cannot have np or tf objects in the json.
    model_params = {
        'learning_rate': model.optimizer.lr.numpy().item(),
        'optimizer': model.optimizer.get_config(),
        'epochs': kwargs.get('epochs', 10),
        'batch_size': default_training_config['batch_size'],
        'monitor_metric': monitor_metric,
        'early_stopping_patience': kwargs.get('early_stopping_patience', 5),
        'checkpoint': kwargs.get('checkpoint', True),
    }
    model_params.update(kwargs)

    # Must recursively convert tensors and numpy arrays to python objects
    def convert_to_python(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.item()
        if isinstance(obj, tf.Tensor):
            return obj.numpy().item()
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        return obj
    
    model_params = convert_to_python(model_params)

    _, dir_params = utils.parse_gcs_path(dir_params)
    utils.json_dump_gcs(model_params, dir_params)

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