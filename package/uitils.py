import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import re

class EarlyStoppingByLossVal(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_classifier_loss', patience=20, restore_best_weights=True):
        super(EarlyStoppingByLossVal, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best = float('inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            raise ValueError(f"Monitor {self.monitor} is not available in logs.")

        if current < self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)


class Metrics_sknet(Callback):

    def __init__(self, validation, acc_threshold, model_save_path):
        super(Metrics_sknet, self).__init__()
        self.validation = validation
        self.acc_threshold = acc_threshold
        self.model_save_path = model_save_path

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracys = []
        self.val_spe = []
        self.best_val_f1 = 0.5
        self.best_model_path = None

    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]
        val_targ = np.argmax(val_targ, axis=1)
        val_predict = self.model.predict(self.validation[0])
        val_predict = tf.argmax(val_predict, axis=1)
        _val_accuracy = np.round(accuracy_score(val_targ, val_predict), 3)
        _val_f1 = np.round(f1_score(val_targ, val_predict), 3)
        _val_recall = np.round(recall_score(val_targ, val_predict), 3)
        _val_precision = np.round(precision_score(val_targ, val_predict), 3)
        _confusion = confusion_matrix(val_targ, val_predict)

        TN = _confusion[0, 0]
        FP = _confusion[0, 1]
        _val_spe = np.round((TN / float(TN + FP)), 3)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_spe.append(_val_spe)
        if _val_accuracy >= self.acc_threshold:
            if _val_f1 >= self.best_val_f1:
                self.best_val_f1 = _val_f1
                print("best f1: {}".format(self.best_val_f1))

                model_save_path = self.model_save_path + f'-acc-{_val_accuracy}-f1-{_val_f1}-pre-{_val_precision}-recall-{_val_recall}-spe-{_val_spe}' + '.h5'
                self.model.save(model_save_path)

                if   self.best_model_path  is  not None and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                self.best_model_path = model_save_path


class Metrics_resnet(Callback):

    def __init__(self, validation, acc_threshold, model_save_path):
        super(Metrics_resnet, self).__init__()
        self.validation = validation
        self.acc_threshold = acc_threshold
        self.model_save_path = model_save_path

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracys = []
        self.val_spe = []
        self.best_val_f1 = 0.5
        self.best_model_path = None  # 跟踪之前最优的模型 便于删除

    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]
        val_targ = np.argmax(val_targ, axis=1)
        val_predict = self.model.predict(self.validation[0])
        val_predict = tf.argmax(val_predict, axis=1)
        _val_accuracy = np.round(accuracy_score(val_targ, val_predict), 3)
        _val_f1 = np.round(f1_score(val_targ, val_predict), 3)
        _val_recall = np.round(recall_score(val_targ, val_predict), 3)
        _val_precision = np.round(precision_score(val_targ, val_predict), 3)
        _confusion = confusion_matrix(val_targ, val_predict)
        TN = _confusion[0, 0]
        FP = _confusion[0, 1]
        _val_spe = np.round((TN / float(TN + FP)), 3)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_spe.append(_val_spe)
        if _val_accuracy >= self.acc_threshold:
            if _val_f1 >= self.best_val_f1:
                self.best_val_f1 = _val_f1
                print("best f1: {}".format(self.best_val_f1))

                model_save_path = self.model_save_path + f'-acc-{_val_accuracy}-f1-{_val_f1}-pre-{_val_precision}-recall-{_val_recall}-spe-{_val_spe}' + '.h5'
                self.model.save_weights(model_save_path)

                if self.best_model_path is not None and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                self.best_model_path = model_save_path


class Metrics_con_resnet(Callback):

    def __init__(self, validation, acc_threshold, model_save_path):
        super(Metrics_con_resnet, self).__init__()
        self.validation = validation
        self.acc_threshold = acc_threshold
        self.model_save_path = model_save_path

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracys = []
        self.val_spe = []
        self.best_val_f1 = 0.5
        self.best_model_path = None

    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]
        val_targ = np.argmax(val_targ, axis=1)
        val_predict = self.model.predict(self.validation[0])[1]
        val_predict = tf.argmax(val_predict, axis=1)
        _val_accuracy = np.round(accuracy_score(val_targ, val_predict), 3)
        _val_f1 = np.round(f1_score(val_targ, val_predict), 3)
        _val_recall = np.round(recall_score(val_targ, val_predict), 3)
        _val_precision = np.round(precision_score(val_targ, val_predict), 3)
        _confusion = confusion_matrix(val_targ, val_predict)
        TN = _confusion[0, 0]
        FP = _confusion[0, 1]
        _val_spe = np.round((TN / float(TN + FP)), 3)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_spe.append(_val_spe)
        if _val_accuracy >= self.acc_threshold:
            if _val_f1 >= self.best_val_f1:
                self.best_val_f1 = _val_f1
                print("best f1: {}".format(self.best_val_f1))

                model_save_path = self.model_save_path + f'-acc-{_val_accuracy}-f1-{_val_f1}-pre-{_val_precision}-recall-{_val_recall}-spe-{_val_spe}' + '.h5'
                self.model.save_weights(model_save_path)

                if self.best_model_path is not None and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                self.best_model_path = model_save_path

class Metrics_con_sknet(Callback):

    def __init__(self, validation, acc_threshold, model_save_path):
        super(Metrics_con_sknet, self).__init__()
        self.validation = validation
        self.acc_threshold = acc_threshold
        self.model_save_path = model_save_path
        self.best_model_path = None

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracys = []
        self.val_spe = []
        self.best_val_f1 = 0.6
        self.best_val_sen = 0.6

    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]
        val_targ = np.argmax(val_targ, axis=1)
        val_predict = self.model.predict(self.validation[0])[1]
        val_predict = tf.argmax(val_predict, axis=1)
        _val_accuracy = np.round(accuracy_score(val_targ, val_predict), 3)
        _val_f1 = np.round(f1_score(val_targ, val_predict), 3)
        _val_recall = np.round(recall_score(val_targ, val_predict), 3)
        _val_precision = np.round(precision_score(val_targ, val_predict), 3)
        _confusion = confusion_matrix(val_targ, val_predict)
        TN = _confusion[0, 0]
        FP = _confusion[0, 1]
        _val_spe = np.round((TN / float(TN + FP)), 3)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_spe.append(_val_spe)
        if _val_accuracy >= self.acc_threshold:
            if _val_f1 >= self.best_val_f1:
                self.best_val_f1 = _val_f1
                print("best f1: {}".format(self.best_val_f1))
                model_save_path = self.model_save_path + f'-acc-{_val_accuracy}-f1-{_val_f1}-pre-{_val_precision}-recall-{_val_recall}-spe-{_val_spe}' + '.h5'
                self.model.save(model_save_path)

                if   self.best_model_path  is  not None and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                self.best_model_path = model_save_path


def plot_confusion_matrix(cm, classes,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    import itertools
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)



def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def calculate(T):

    TN, TP, FP, FN = T[0, 0], T[1, 1], T[0, 1], T[1, 0]
    Sen = TP / (TP + FN + 1e-6)
    Spe = TN / (TN + FP + 1e-6)
    Pre = TP / (TP + FP + 1e-6)
    F1 = 2 * TP / (2 * TP + FP + FN)
    Acc = (TN + TP) / (TN + TP + FP + FN)

    return Acc, Sen, Spe, Pre, F1



def find_best_model1(model_path, i):
    model_pattern = re.compile(r"fold_{}-acc-[\d.]+-f1-([\d.]+)-pre-[\d.]+-recall-([\d.]+)-spe-[\d.]+\.h5".format(i))
    best_f1_score = 0
    best_recall_score = 0
    best_model_path = None

    for model_file in os.listdir(model_path):
        match = model_pattern.match(model_file)
        if match:
            f1_score = float(match.group(1))
            recall_score = float(match.group(2))
            if recall_score > best_recall_score or (recall_score == best_recall_score and f1_score > best_f1_score):
                best_f1_score = f1_score
                best_recall_score = recall_score
                best_model_path = os.path.join(model_path, model_file)
    return best_model_path


def find_best_model2(model_path, i):

    model_pattern = re.compile(r"fold_{}-acc-[\d.]+-f1-([\d.]+)-pre-[\d.]+-recall-([\d.]+)-spe-[\d.]+\.h5".format(i))
    best_f1_score = 0
    best_recall_score = 0
    best_model_path = None
    for model_file in os.listdir(model_path):
        match = model_pattern.match(model_file)
        if match:
            f1_score = float(match.group(1))
            recall_score = float(match.group(2))
            if f1_score > best_f1_score or (f1_score == best_f1_score and recall_score > best_recall_score):
                best_f1_score = f1_score
                best_recall_score = recall_score
                best_model_path = os.path.join(model_path, model_file)
    return best_model_path
