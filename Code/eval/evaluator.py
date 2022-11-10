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

    @staticmethod
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


    @staticmethod
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
        The model dir should contain:
        - model/
            - saved model files
        - [logdir]/
            - train/
                - events.out.tfevents.*
            - val/
                - events.out.tfevents.*
        Args:
            model_dir (str): The path to the model directory.
        """
        self.model_dir = model_dir
        self.is_google_cloud = 'gs://' in model_dir

    def plot_history(self, history, metric='loss'):
        """
        Plot the training history of a model.

        Args:
            metric (str): The metric to plot. Defaults to 'loss'.
        """
        if metric not in ['loss', 'accuracy']:
            raise ValueError(f'Unsupported metric: {metric}')
        plt.clf()
        history = self.load_history()
        plt.plot(history[metric])
        plt.plot(history[f'val_{metric}'])
        plt.title(f'Model {metric}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # plt.plot(history.history[metric])
        # plt.plot(history.history[f'val_{metric}'])
        # plt.title(f'Model {metric}')
        # plt.ylabel(metric)
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
    
    def load_history(self):
        """
        Load the training history of a model from tensorboard logs.
        """
        train_logdir = os.path.join(self.model_dir, 'train')
        val_logdir = os.path.join(self.model_dir, 'val')
        if self.is_google_cloud:
            train_logdir = utils.download_dir(train_logdir)
            val_logdir = utils.download_dir(val_logdir)
        train_events = glob.glob(os.path.join(train_logdir, '*'))
        val_events = glob.glob(os.path.join(val_logdir, '*'))
        train_events.sort()
        val_events.sort()
        train_events = [tf.compat.v1.train.summary_iterator(event) for event in train_events]
        val_events = [tf.compat.v1.train.summary_iterator(event) for event in val_events]
        train_history = {}
        val_history = {}
        for event in train_events:
            for value in event:
                for v in value.summary.value:
                    if v.tag not in train_history:
                        train_history[v.tag] = []
                    train_history[v.tag].append(v.simple_value)
        for event in val_events:
            for value in event:
                for v in value.summary.value:
                    if v.tag not in val_history:
                        val_history[v.tag] = []
                    val_history[v.tag].append(v.simple_value)
        history = {}
        for k in train_history:
            if k not in val_history:
                continue
            history[k] = train_history[k]
            history[f'val_{k}'] = val_history[k]
        return history
