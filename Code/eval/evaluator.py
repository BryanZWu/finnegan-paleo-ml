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
        self.is_google_cloud = 'gs://' in model_dir
        if not self.is_google_cloud:
            self.model_dir = model_dir
        else:
            # Get the bucket name and the path to the model directory
            self.bucket_name, self.model_dir = utils.parse_gcs_path(model_dir)


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
    
    def load_history(self, date=None):
        """
        Load the training history of a model from tensorboard logs.

        Args:
            date (str): The date string. If None, the most recent date will be used.
                Format: YYYY-MM-DD.HH-MM-SS
        """
        if date is None:

            if self.is_google_cloud:
                training_logs = utils.ls_cloud_dir(self.model_dir, self.bucket_name)
            else:
                training_logs = glob.glob(os.path.join(self.model_dir, '*'))
            
            # remove in progress logs and non-logs (aka no date string)
            log_filter = lambda log: re.match(r'^\d{4}-\d{2}-\d{2}\.\d{2}-\d{2}-\d{2}/', log)
            training_logs = [log for log in training_logs if log_filter(log)]
            
            # sort logs by date, which is just lexographically
            training_logs.sort()
            # get the most recent log. Format: YYYY-MM-DD.HH-MM-SS/
            training_logs = training_logs[-1]

            train_logdir = os.path.join(self.model_dir, training_logs, 'train')
            val_logdir = os.path.join(self.model_dir, training_logs, 'validation')

        else:
            train_logdir = os.path.join(self.model_dir, date, 'train')
            val_logdir = os.path.join(self.model_dir, date, 'validation')
            train_events = glob.glob(os.path.join(train_logdir, '*'))
            val_events = glob.glob(os.path.join(val_logdir, '*'))
        
        if self.is_google_cloud:
            # train_logdir = utils.download_dir(train_logdir, self.bucket_name)
            # val_logdir = utils.download_dir(val_logdir, self.bucket_name)
            
            # tf2 requires tf.data.TFRecordDataset to read the logs. This means 
            # cloud only if TPU is used.

            train_events = utils.ls_cloud_dir(train_logdir, self.bucket_name)
            val_events = utils.ls_cloud_dir(val_logdir, self.bucket_name)


        if len(train_events) == 0:
            raise FileNotFoundError(f'No training logs found in {train_logdir}')
        if len(val_events) == 0:
            raise FileNotFoundError(f'No validation logs found in {val_logdir}')

        train_events.sort()
        val_events.sort()
        '''
        TODO adapt the following code from https://github.com/tensorflow/tensorboard/issues/2745
        from tensorflow.core.util import event_pb2

        serialized_examples = tf.data.TFRecordDataset(path)
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            for value in event.summary.value:
            t = tf.make_ndarray(value.tensor)
            print(value.tag, event.step, t, type(t))
        '''
        get_full_path = lambda event: utils.get_gcs_path(os.path.join(train_logdir, event), self.bucket_name)

        # Do the following on CPU instead of TPU
        with tf.distribute.OneDeviceStrategy('CPU').scope():
            train_events = [tf.data.TFRecordDataset(get_full_path(event)) for event in train_events]
            val_events = [tf.data.TFRecordDataset(get_full_path(event)) for event in val_events]

            train_history = {}
            val_history = {}

            for event in train_events:
                for serialized_data in event:
                    try:
                        event = event_pb2.Event.FromString(serialized_data.numpy())
                        for value in event.summary.value:
                            if value.tag not in train_history:
                                train_history[value.tag] = []
                                print(value.tag)
                                print(tf.make_ndarray(value.tensor))
                                # train_history[value.tag].append(value.simple_value)
                    except tf.errors.OutOfRangeError:
                        break
        return



        train_events = [tf.compat.v1.train.summary_iterator(event) for event in train_events]
        val_events = [tf.compat.v1.train.summary_iterator(event) for event in val_events]
        train_history = {}
        val_history = {}
        for event in train_events:
            for e in event:
                for v in e.summary.value:
                    if v.tag not in train_history:
                        train_history[v.tag] = []
                    # print the tensor 
                    print(dir(v))
                    print(v)
                    print(v.tag)
                    print(v.simple_value)
                    break
                    train_history[v.tag].append(v.simple_value)
        for event in val_events:
            for e in event:
                for v in e.summary.value:
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
