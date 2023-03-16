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
    
    @staticmethod
    def predict_y(model, dataset, batch_size=128, target='species'):
        """
        Predict labels for a given set of inputs.

        Args:
            model (keras.Model): The model to use for prediction.
            dataset (tf.data.Dataset): The dataset to predict labels for.

        Returns:
            A tuple of (y_true, y_pred).
        """
        model_output_inds = {'species': (0, -1), 'chamber_broken': (-1, None)}
        if target not in model_output_inds:
            raise ValueError('Invalid target: {}'.format(target))

        y_true = []
        y_pred = []
        for item in dataset.batch(batch_size):
            y_true.append(item[target])
            pred = model.predict(item['image'])
            y_pred.append(pred[:, model_output_inds[target][0]:model_output_inds[target][1]])
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        # y_true = y_true.reshape(-1)
        # y_pred = y_pred.reshape(-1)
        return y_true, y_pred
    
    @staticmethod
    def get_y_ord_labels(y_pred, ind_range):
        """
        Get the ordinal labels for a set of predicted labels.

        Args:
            y_pred (np.array): The predicted labels.
            ind_range int[2]: the start and end indices of the ordinal labels.

        Returns:
            A list of ordinal labels.
        """
        y_pred_ord = np.argmax(y_pred[:, ind_range[0]:ind_range[1]], axis=1)
        return y_pred_ord

class PlottingEvaluator(Evaluator):
    """
    Takes output from a model and plots visualizations related to the 
    model's performance.
    """

    def generate_plots(self, species_list=None, plots='all'):
        """
        Generate plots for a given set of true and predicted labels.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
            plots (list): A list of plots to generate. Defaults to 'all'.
        """
        print(self, plots)
        if plots == 'all':
            plots = ['confusion_matrix', 'roc_curve', 'precision_recall_curve']
        y_true = self.y_true
        y_pred = self.y_pred
        return {plot: PlottingEvaluator.generate_plot(y_true, y_pred, plot, species_list=species_list) for plot in plots}
    
    @staticmethod
    def generate_plot(y_true, y_pred, plot, species_list, **kwargs):
        """
        Generate a plot for a given set of true and predicted labels.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
            plot (str): The plot to generate.

        Returns:
            A matplotlib figure.
        """
        if plot == 'confusion_matrix':
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
            return PlottingEvaluator.confusion_matrix(cm, species_list, **kwargs)
        elif plot == 'roc_curve':
            return PlottingEvaluator.roc_curve(y_true, y_pred, species_list, **kwargs)
        elif plot == 'precision_recall_curve':
            return PlottingEvaluator.precision_recall_curve(y_true, y_pred, species_list, **kwargs)
        else:
            raise ValueError(f'Invalid plot: {plot}')


    @staticmethod
    def confusion_matrix(cm, class_names=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
    
    @staticmethod
    def roc_curve(y_true, y_pred, class_names, title='ROC curve'):
        """
        Plot the ROC curve for a given set of true and predicted labels.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
            class_names (list): A list of class names.
            title (str): The title of the plot.
        """
        plt.clf()
        plt.title(title)
        for i in range(len(class_names)):
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true[:, i], y_pred[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (\
                AUC = {sklearn.metrics.auc(fpr, tpr):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='best')
        return plt.gcf()

    @staticmethod
    def precision_recall_curve(y_true, y_pred, class_names, title='Precision-Recall curve'):
        """
        Plot the precision-recall curve for a given set of true and predicted labels.

        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The predicted labels.
            class_names (list): A list of class names.
            title (str): The title of the plot.
        """
        plt.clf()
        plt.title(title)
        for i in range(len(class_names)):
            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true[:, i], y_pred[:, i])
            plt.plot(recall, precision, label=f'{class_names[i]} (\
                AUC = {sklearn.metrics.auc(recall, precision):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='best')
        return plt.gcf()


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


    def plot_history(self, train_history, val_history, metric='loss'):
        """
        Plot the training history of a model.

        Args:
            metric (str): The metric to plot. Defaults to 'loss'.
        """
        allowed_metrics = {
            'loss': ['loss', 'val_loss', 'epoch_loss'],
            'accuracy': ['accuracy', 'val_accuracy', 'epoch_accuracy'],
            'precision': ['precision', 'val_precision', 'epoch_precision'],
            'recall': ['recall', 'val_recall', 'epoch_recall'],
            'f1_score': ['f1_score', 'val_f1_score', 'epoch_f1_score'],
        }
        if metric not in allowed_metrics:
            raise ValueError(f'Unsupported metric: {metric}')

        key = None
        # Find the specific key in the history dict
        for k in allowed_metrics[metric]:
            if k in train_history:
                key = k
                break
        if key is None:
            raise ValueError(f'Could not find metric {metric} in history')
        assert key in val_history, f'Could not find metric {metric} in val history'

        
        metric = key

        plt.clf()
        plt.plot(train_history[metric])
        plt.plot(val_history[metric])
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

        ## Get the most recent log directory ##
        if date is None:
            if self.is_google_cloud:
                training_logs = utils.ls_cloud_dir(self.model_dir, self.bucket_name)
            else:
                training_logs = glob.glob(os.path.join(self.model_dir, '*'))
            
            # remove in progress logs and non-logs (aka no date string)
            log_filter = lambda log: re.match(r'^\d{4}-\d{2}-\d{2}\.\d{2}-\d{2}-\d{2}/', log)

            unfiltered_logs = training_logs
            training_logs = [log for log in training_logs if log_filter(log)]
            
            # sort logs by date, which is just lexographically
            training_logs.sort()
            if len(training_logs) == 0:
                if len(unfiltered_logs) == 0:
                    raise ValueError(f'No logs found. Check your specified model directory: {self.model_dir}')
                else:
                    raise ValueError(f'No valid logs in {unfiltered_logs}')
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


        ## Load the logs into TFRecordDataset ##

        train_events.sort()
        val_events.sort()
        # code adapted from from https://github.com/tensorflow/tensorboard/issues/2745
        def get_full_path(event, val=False):
            logdir = val_logdir if val else train_logdir
            path = os.path.join(logdir, event)
            return utils.get_gcs_path(path, self.bucket_name)

        # Use only one TPU core
        with tf.distribute.OneDeviceStrategy(tf.config.list_logical_devices('TPU')[0]).scope():
            train_events = [tf.data.TFRecordDataset(get_full_path(event)) for event in train_events]
            val_events = [tf.data.TFRecordDataset(get_full_path(event, val=True)) for event in val_events]

            train_history = {}
            val_history = {}

            step = 0

            tags = set(['epoch_loss', 'epoch_accuracy', 'epoch_precision'])

            for train_event in train_events:
                for serialized_data in train_event:
                    event = event_pb2.Event.FromString(serialized_data.numpy())
                    for value in event.summary.value:
                        assert event.step >= step, f'Event step {event.step} is less than previous step {step}'
                        if value.tag not in tags:
                            continue
                        if value.tag not in train_history:
                            train_history[value.tag] = []
                            train_history[value.tag].append(tf.make_ndarray(value.tensor).item())
                        else:
                            train_history[value.tag].append(tf.make_ndarray(value.tensor).item())
            for val_event in val_events:
                for serialized_data in val_event:
                    event = event_pb2.Event.FromString(serialized_data.numpy())
                    for value in event.summary.value:
                        if value.tag not in tags:
                            continue
                        assert event.step >= step, f'Event step {event.step} is less than previous step {step}'
                        if value.tag not in val_history:
                            val_history[value.tag] = []
                            val_history[value.tag].append(tf.make_ndarray(value.tensor).item())
                        else:
                            val_history[value.tag].append(tf.make_ndarray(value.tensor).item())
        self.train_history = train_history
        self.val_history = val_history
        return train_history, val_history