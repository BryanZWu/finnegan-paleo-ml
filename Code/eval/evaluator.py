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


class GeneraExtractor:
    @staticmethod
    def get_genera(species_names):
        '''
        Get a list of genera from a list of species names.
        Returns a list of genera, a dictionary mapping genera to indices, and a dictionary mapping species indices to genus indices.
        '''
        n_genera_found = 0
        genera_set = {} # Note that dictionaries in pyhon 3.7+ preserve insertion order
        genera = []
        # A dict from species index to genus index
        ind_to_genus = {}
        for i, species in enumerate(species_names):
            genus = species.split(' ')[0]
            if genus not in genera_set:
                genera_set[genus] = n_genera_found
                n_genera_found += 1
            genera.append(genera_set[genus])
            ind_to_genus[i] = genera_set[genus]
        return genera, genera_set, ind_to_genus
    
    @staticmethod
    def convert_to_genera(y_true, y_pred, ind_to_genus, logit=True):
        '''
        Convert a list of species indices to a list of genus indices.

        Args:
            y_true (np.array): An array of true species indices, shape (n_samples,).
            y_pred (tf.Tensor): An array of predicted species probabilities/logits, shape (n_samples, n_classes).
            ind_to_genus (dict): A dictionary mapping species indices to genus indices.
            logit (bool): Whether y_pred is a tensor of logits or probabilities.
        
        Returns:
            y_true_genera (np.array): An array of true genus indices, shape (n_samples,).
            y_pred_genera (np.array): An array of predicted genus probabilities, shape (n_samples, n_genera).
        '''
        if logit:
            # Softmax to get probabilities
            y_pred = softmax(y_pred, axis=1)
        y_true_genera = []
        y_pred_genera = []
        n_samples, _ = y_pred.shape
        for i in range(n_samples):
            y_true_genus = ind_to_genus[y_true[i]]
            y_true_genera.append(y_true_genus)

            # Now, convert (n_samples, n_classes) to (n_samples, n_genera) by adding 
            # according to the genus index
            y_pred_genus = np.zeros(len(ind_to_genus))
            for j in range(num_classes):
                y_pred_genus[ind_to_genus[j]] += y_pred[i][j]
            y_pred_genera.append(y_pred_genus)
        
        y_true_genera = np.array(y_true_genera)
        y_pred_genera = np.array(y_pred_genera)
        return y_true_genera, y_pred_genera


class Evaluator:
    def __init__(self, y_trues, y_preds):
        """
        Initialize the MetricEvaluator.

        Args:
            y_trues ([np.array]): A list of true labels.
            y_preds ([np.array]): A list of predicted labels.
        """
        self.y_trues = y_trues
        self.y_preds = y_preds
    
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
    Code cleanup of the old plotting evaluator.
    """
    def confusion_matrix(self, target_ind, **kwargs):
        """
        Generate a confusion matrix.
        
        Args:
            target_ind (int): the index of the y in y_trues and y_preds to use.
        """
        default_args = {
            'normalize': False,
            'title': 'Confusion matrix',
            'cmap': plt.cm.Blues,
            'interpolation': 'nearest',
            'annot': True,
            'xticklabels': 'auto',
            'yticklabels': 'auto',
            }
        kwargs = {**default_args, **kwargs}
        # >1 targets = argmax, 1 target = sigmoid logit = >0 
        get_ord_labels = lambda y: y.argmax(axis=1) if y.shape[1] > 1 else y > 0
        cm = sklearn.metrics.confusion_matrix(
            self.y_trues[target_ind],
            get_ord_labels(self.y_preds[target_ind]),
            normalize=kwargs['normalize'],
        )
        sns.heatmap(cm, annot=kwargs['annot'], cmap=kwargs['cmap'], fmt='.2f', xticklabels=kwargs['xticklabels'], yticklabels=kwargs['yticklabels'])
        plt.title(kwargs['title'])
        return plt.gcf()
    
    def genera_confusion_matrix(self, species_names, **kwargs):
        """
        Instead of a species level confusion matrix, generate a genera level
        confusion matrix which is done by taking and grouping the species
        by the first word in the species name.

        Only works
        """
        return self.confusion_matrix(0, **kwargs)



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