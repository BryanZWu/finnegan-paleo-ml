
from inspect import signature
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.constants import *
from common.imports import *
from common import utils

# Provides custom losses and metrics for multitarget training.

def species_broken_loss(y_true, y_pred, ratio=1):
    """
    Multi-task loss function for the species and and whether the shell is broken.

    Args:
        y_true: the true labels
        y_pred: the predicted labels
        ratio: the ratio of the two losses. The species loss is weighted by this ratio.
    """
    # Species is categorical crossentropy on the first n-1 columns.
    # Shell broken is binary crossentropy on the last column.

    # print('species loss inputs shape: ', y_true[:, :-1].shape, y_pred[:, :-1].shape)
    species_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true[:, :-1], y_pred[:, :-1], from_logits=True)
    # print('species_loss: ', species_loss.shape)
    # print('broken loss inputs shape: ', y_true[:, -1].shape, y_pred[:, -1].shape)
    broken_loss = tf.keras.losses.binary_crossentropy(y_true[:, -1], y_pred[:, -1], from_logits=True)
    # print('broken_loss: ', broken_loss.shape)
    # print()
    # print('broken_loss: ', broken_loss)
    # print('species_loss: ', species_loss)
    # print('combined_loss: ', species_loss * ratio + broken_loss)
    # print(K.print_tensor(species_loss * ratio + broken_loss, message="loss: "))
    #print the loss
    return species_loss * ratio + broken_loss

def SpeciesBrokenLoss(ratio=1):
    """
    Wrapper for species_broken_loss to allow it to be used as a loss function.
    """
    return lambda y_true, y_pred: species_broken_loss(y_true, y_pred, ratio)

def species_accuracy(y_true, y_pred):
    """
    Accuracy for the species task.
    """
    return tf.keras.metrics.sparse_categorical_accuracy(y_true[:, :-1], y_pred[:, :-1])

def broken_accuracy(y_true, y_pred):
    """
    Accuracy for the broken task.
    """
    return tf.keras.metrics.binary_accuracy(y_true[:, -1], y_pred[:, -1])

def species_broken_accuracy(y_true, y_pred, binary_threshold=0.5):
    """
    The accuracy of getting both the species and whether the shell is broken correct.
    """
    # acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
    # In keras:
    broken_true, broken_pred = y_true[:, -1], y_pred[:, -1]
    species_true, species_pred = y_true[:, 0], y_pred[:, :-1]
    
    broken_acc = tf.math.equal(tf.cast(broken_true, tf.bool), tf.math.greater(broken_pred, binary_threshold))
    species_acc = tf.math.equal(tf.cast(species_true, tf.int64), tf.math.argmax(species_pred, axis=1))

    broken_acc, species_acc = tf.squeeze(broken_acc), tf.squeeze(species_acc)

    assert broken_acc.shape == species_acc.shape, f'broken_acc.shape: {broken_acc.shape}, species_acc.shape: {species_acc.shape}'

    return tf.math.reduce_mean(tf.cast(tf.math.logical_and(broken_acc, species_acc), tf.float32))

