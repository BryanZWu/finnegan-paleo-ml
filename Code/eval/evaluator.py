"""
Centralized file for model evaluation.

Supported evluation metrics:
- Precision
- Recall
- F1 score
- Accuracy

Supported visualizations:
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Training history--loss and accuracy of training and validation sets
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.constants import *
from common.imports import *
from common import utils

class Evaluator:
    def __init__(self, y_true, y_pred):
        """
        Initialize the MetricEvaluator.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
        """
        self.y_true = y_true
        self.y_pred = y_pred


class MetricEvaluator(Evaluator):
    '''
    Evaluates a set of numerical metrics
    '''

    metric_map = {
        'precision': tf.keras.metrics.Precision,
        'recall': tf.keras.metrics.Recall,
        'accuracy': tf.keras.metrics.Accuracy,
        'f1': tfa.metrics.F1Score
    }

    def calculate_metrics(self, metrics='all'):
        """
        Calculate a set of metrics for a given set of true and predicted labels.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
            metrics (list): A list of metrics to calculate. Defaults to 'all'.
        """
        if metrics == 'all':
            metrics = ['precision', 'recall', 'f1', 'accuracy']
        return {metric: self.calculate_metric(metric) for metric in metrics}

    def calculate_metric(self, metric):
        """
        Calculate a metric for a given set of true and predicted labels.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
            metric (str): The metric to calculate.
        """
        if metric not in self.metric_map:
            raise ValueError(f'Unsupported metric: {metric}')
        metric_fn = self.metric_map[metric]()
        metric_fn.update_state(self.y_true, self.y_pred)
        return metric_fn.result().numpy()

class PlottingEvaluator(Evaluator):
    """
    Takes output from a model and plots visualizations related to the 
    model's performance.
    """

    def generate_plots(y_true, y_pred, plots='all'):
        """
        Generate plots for a given set of true and predicted labels.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
            plots (list): A list of plots to generate. Defaults to 'all'.
        """
        if plots == 'all':
            plots = ['confusion_matrix', 'roc_curve', 'precision_recall_curve']
        return {plot: generate_plot(y_true, y_pred, plot) for plot in plots}


    def confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        Args:
            cm (np.array): The confusion matrix to plot.
            class_names (list): A list of class names.
            normalize (bool): Whether to normalize the confusion matrix.
            title (str): The title of the plot.
            cmap (matplotlib.cm): The color map to use.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))

class HistoryVisualizer:
    """
    Visualizes the training history of a model.
    """

    def __init__(self, model_dir):
        """
        Initialize the HistoryVisualizer.

        Args:
            model_dir (str): The path to the model directory.
        """
        self.model_dir = model_dir

    def plot_history(history, metric='loss'):
        """
        Plot the training history of a model.

        Args:
            history (tf.keras.callbacks.History): The training history of a model.
            metric (str): The metric to plot. Defaults to 'loss'.
        """
        if metric not in ['loss', 'accuracy']:
            raise ValueError(f'Unsupported metric: {metric}')
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'Model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()