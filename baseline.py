from sklearn.preprocessing import LabelEncoder
import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from package.augmentations import  smoothed_filter
from tensorflow import keras
from  package.savefigure_base import savefigure
import random as python_random
import tensorflow as tf
import pandas as pd
from package.uitils import Metrics_sknet, calculate, Metrics_resnet, find_best_model2
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.models import load_model
from models.sknet2 import ts_model_ours, Axpby, ts_model
from package.load_newdata import load_data1
from models.resnet2 import resnet
from package.config import data_name, data_ahi, label_best
parser = argparse.ArgumentParser()
# --------------------------- Model parameters ---------------------------
parser.add_argument('--path', default=r'.\results', type=str)
parser.add_argument('--net_name', default=r'test', type=str)
parser.add_argument('--basemodel', default='resnet', type=str)              # sknet or resnet
parser.add_argument('--data_path', default=r'.\data\label_data_5min.csv', type=str)
parser.add_argument('--epoch', default=60, type=int)
parser.add_argument('--fold', default=5, type=int, help='n-fold')
parser.add_argument('--gpu', default=1, type=int, help='gpu NO.x')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--learning_rate', default=0.0015, type=float)
parser.add_argument('--acc_threshold', default=0.77, type=float)
parser.add_argument('--seed', default=123, type=int)
args = parser.parse_args()
# ------------------------------------------------------------------------
label_use=[item - 84 for item in label_best]
data_label=np.array(label_use)
data_name=np.array(data_name)
data_ahi=np.array(data_ahi)
delete_hdf5_flag = True
stand_flag = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_model = {
    "ts": ts_model,
    "ts_ours": ts_model_ours,
    "resnet": resnet, }

def my_main(train_data_name,val_data_name, test_data_name, data_all_path, result, model_path, batch_size, epoch,
             learning_rate,  acc_threshold,  k ,summary, basemodel):

    data_all = pd.read_csv(data_all_path)
    train_data = data_all[data_all['id'].isin(train_data_name)]
    test_data = data_all[data_all['id'].isin(test_data_name)]
    val_data = data_all[data_all['id'].isin(val_data_name)]

    X_train, X_valid, X_test, SA_train, SA_val, SA_test,train_mean,train_std = load_data1(train_data, val_data, test_data)
    SA_test1 = tf.argmax(SA_test, axis=1)
    if basemodel == 'ts':
        model = ts_model(input_shape=(500, 1))
        model.summary()
        metrics = Metrics_sknet(validation=(X_valid, SA_val), acc_threshold=acc_threshold, model_save_path=model_path)

    elif basemodel == 'ts_ours':
        model = ts_model_ours(input_shape=(500, 1))
        model.summary()
        metrics = Metrics_sknet(validation=(X_valid, SA_val), acc_threshold=acc_threshold, model_save_path=model_path)

    else:
        model = resnet()
        model.build(input_shape=(1, 500, 1))
        model.summary()
        metrics = Metrics_resnet(validation=(X_valid, SA_val), acc_threshold=acc_threshold, model_save_path=model_path)

    model.compile(  loss='binary_crossentropy',
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999),
                    metrics=["accuracy"])


    X_trains = smoothed_filter(X_train)
    X_train = tf.concat([X_train, X_trains], axis=0)
    SA_train = tf.concat([SA_train, SA_train], axis=0)

    early_stopping = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    history = model.fit(X_train, SA_train,  epochs=epoch, verbose=2,
                           validation_data=(X_valid, SA_val),
                           callbacks=[metrics, early_stopping],
                           batch_size=batch_size,  shuffle=True)

    best_model = find_best_model2(result, k)
    print(f"The best model path is: {best_model}")
    if best_model is not None:
        if basemodel == 'resnet':
            model.load_weights(best_model)
        else:
            model = load_model(best_model, custom_objects={'Axpby': Axpby})

    score = model.evaluate(X_test, SA_test , batch_size=512)
    loss = round(score[0], 3)
    test_predict_raw = model.predict(X_test)
    test_predict = np.argmax(test_predict_raw, axis=1)
    test_targ = SA_test1
    savefigure(history, result, test_targ, test_predict,test_predict_raw[:,1], k )
    fpr, tpr, thresholds = roc_curve(test_targ, test_predict_raw[:,1])
    Auc = auc(fpr, tpr)
    ta = confusion_matrix(test_targ, test_predict)
    Acc, Sen, Spe, Pre, F1 = calculate(ta)

    with open(summary, "a") as f:
        f.write('acc-{:.3f}-sen-{:.3f}-spe-{:.3f}-pre-{:.3f}-f1-{:.3f}-auc{:.3f}-loss{:.3f}'.format(Acc, Sen, Spe, Pre, F1, Auc, loss) + '\n')

    ahi = []
    pre_ahi = []
    for i in  test_data_name:
        test_data = data_all[data_all['id'].isin([i])]
        test_PPI = test_data['RRdata']
        test_PPI = list(map(eval, test_PPI))
        test_PPI = np.array(test_PPI[:])
        label = LabelEncoder().fit(test_data.label_SA)
        label_sa = label.transform(test_data.label_SA)

        test_PPI = (test_PPI - train_mean) / train_std
        test_PPI = test_PPI.reshape(test_PPI.shape[0], 500, 1)
        test_PPI = test_PPI.astype('float32')

        test_predict = model.predict(test_PPI)
        test_targ = label_sa
        test_predict = np.argmax(test_predict, axis=1)
        ahi.append((sum(test_targ) * 60 / len(test_targ)))
        pre_ahi.append((sum(test_predict) * 60 / len(test_predict)))

    ahi = np.array(ahi)
    pre_ahi = np.array(pre_ahi)
    mae = np.sum(np.abs(ahi - pre_ahi)) / len(ahi)
    targ = np.where(ahi >= 30, 1, 0)
    pred = np.where(pre_ahi >= 30, 1, 0)
    ta = confusion_matrix(targ, pred)
    acc, sen, spe, pre, f1 = calculate(ta)

    mean_ahi = tf.reduce_mean(ahi)
    mean_pre_ahi = tf.reduce_mean(pre_ahi)
    std_ahi = tf.sqrt(tf.reduce_mean(tf.square(ahi - mean_ahi)))
    std_pre_ahi = tf.sqrt(tf.reduce_mean(tf.square(pre_ahi - mean_pre_ahi)))
    covariance = tf.reduce_mean((ahi - mean_ahi) * (pre_ahi - mean_pre_ahi))
    pcc = covariance / (std_ahi * std_pre_ahi)
    return Acc, Sen, Spe, Pre, F1, acc, sen, spe, pre, f1, mae, pcc, Auc


if __name__ == '__main__':

    result = args.path + '/' + args.net_name
    summary = result + '/summary.txt'
    if not os.path.exists(result):
        os.makedirs(result)
    data_path = args.data_path
    fold_n = args.fold
    Acc, Sen, Spe, Pre, F1, Auc = [np.zeros((1, fold_n)) for _ in range(6)]
    acc, sen, spe, pre, f1, mae, pcc = [np.zeros((1, fold_n)) for _ in range(7)]
    gpu = args.gpu
    epoch = args.epoch
    net_name = args.net_name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    seed = args.seed
    acc_threshold = args.acc_threshold
    basemodel = args.basemodel
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu)
    k = 0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
    for train_val_index, test_index in kfold.split(data_name, data_label):
        train_val_data, test_data_name = data_name[train_val_index], data_name[test_index]
        train_val_label, test_label = data_label[train_val_index], data_label[test_index]
        train_val_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
        for train_index, val_index in train_val_kfold.split(train_val_data, train_val_label):
            train_data_name, val_data_name = train_val_data[train_index], train_val_data[val_index]

            model_path = result + '/fold_' + str(k)
            Acc[0, k], Sen[0, k], Spe[0, k], Pre[0, k], F1[0, k], acc[0, k], sen[0, k], spe[0, k], pre[0, k], f1[0, k], mae[0, k], pcc[0, k], Auc[0, k] = \
                my_main(train_data_name, val_data_name, test_data_name, data_path, result, model_path, batch_size, epoch,
                        learning_rate, acc_threshold,  k, summary, basemodel)
            k = k + 1
            break
    with open(summary, "a") as f:
        f.write("segment detection mean acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} Auc:{:.4f}".format(
            np.mean(Acc), np.mean(Sen), np.mean(Spe), np.mean(Pre), np.mean(F1), np.mean(Auc)) + '\n')
        f.write("segment detection std  acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} Auc:{:.4f} ".format(
            np.std(Acc), np.std(Sen), np.std(Spe), np.std(Pre), np.std(F1), np.std(Auc)) + '\n')

        f.write("individual detection mean acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} MAE:{:.4f} PCC:{:.4f}".format(
            np.mean(acc), np.mean(sen), np.mean(spe), np.mean(pre), np.mean(f1), np.mean(mae) , np.mean(pcc))  + '\n')
        f.write("individual detection std  acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} MAE:{:.4f} PCC:{:.4f}".format(
            np.std(acc), np.std(sen), np.std(spe), np.std(pre), np.std(f1), np.std(mae) , np.std(pcc))  + '\n')
