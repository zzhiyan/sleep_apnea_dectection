# 2025/9/21
# -----------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from package.augmentations import  smoothed_filter
from tensorflow import keras
from  package.savefigure import savefigure
import random as python_random
import tensorflow as tf
import pandas as pd
from package.uitils import Metrics_con_sknet,Metrics_con_resnet, calculate, find_best_model2,  EarlyStoppingByLossVal
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.models import load_model
from models.loss import  sup_obs
from package.load_newdata import load_data2
from models.sknet2 import hybrid_ts_model_ours, Axpby, hybrid_ts_model
from models.resnet2 import hybrid_resnet_model
from package.config import data_name, data_ahi, label_best
parser = argparse.ArgumentParser()
# --------------------------- Model parameters ---------------------------
parser.add_argument('--path', default=r'.\results', type=str)
parser.add_argument('--net_name', default=r'test', type=str)
parser.add_argument('--basemodel', default='resnet', type=str)
parser.add_argument('--epoch', default=6, type=int)
parser.add_argument('--fold', default=5, type=int, help='n-fold')
parser.add_argument('--gpu', default=2, type=int, help='gpu NO.x')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--learning_rate', default=0.0015, type=float)
parser.add_argument('--acc_threshold', default=0.78, type=float)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--a', default=0.15, type=float)
args = parser.parse_args()

base_model = {
    "ts": hybrid_ts_model,
    "ts_ours": hybrid_ts_model_ours,
    "resnet": hybrid_resnet_model, }

# ------------------------------------------------------------------------
label_use=[item - 84 for item in label_best]
data_label=np.array(label_use)
data_name=np.array(data_name)
data_ahi=np.array(data_ahi)
delete_hdf5_flag = True
stand_flag = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def my_main(train_data_name,val_data_name, test_data_name, data_all_path, result, model_path, batch_size, epoch,
             learning_rate,  acc_threshold,  k, summary, basemodel, a):

    data_all = pd.read_csv(data_all_path)

    filtered_data = data_all[(data_all['id'] == 'PPG_ID_69') & (data_all['label_SA'] == 1)]
    row = filtered_data.iloc[179]  # Typical SA segment
    entropy = eval(row['entropy'])
    entropy = np.array(entropy).reshape(1, -1)
    entropy_sum = eval(row['entropy_sum'])
    entropy_sum = np.array(entropy_sum).reshape(1, -1)
    sort = eval(row['sort'])
    sort = np.array(sort).reshape(1, -1)
    e1 = tf.concat([entropy, entropy_sum, sort], axis=-1)

    filtered_data = data_all[(data_all['id'] == 'PPG_ID_116') & (data_all['label_SA'] == 0)]
    row = filtered_data.iloc[20]  # Typical NSA segment
    entropy = eval(row['entropy'])
    entropy = np.array(entropy).reshape(1, -1)
    entropy_sum = eval(row['entropy_sum'])
    entropy_sum = np.array(entropy_sum).reshape(1, -1)
    sort = eval(row['sort'])
    sort = np.array(sort).reshape(1, -1)
    e0 = tf.concat([entropy, entropy_sum, sort], axis=-1)

    # Filter data where the 'id' is 'PPG_ID_69' and the 'label_SA' is 1.
    condition = (data_all['id'] == 'PPG_ID_69') &  (data_all['label_SA'] == 1)
    filtered_data = data_all[condition]
    # Retrieve the index for row 179 where the 'id' is 'PPG_ID_69' and 'label_SA' is 1.
    index_to_drop = filtered_data.index[179]
    data_all = data_all.drop(index_to_drop)  # delete

    # Filter data where the 'id' is 'PPG_ID_116' and the 'label_SA' is 0.
    condition = (data_all['id'] == 'PPG_ID_116') & (data_all['label_SA'] == 0)
    filtered_data = data_all[condition]
    index_to_drop = filtered_data.index[20]
    data_all = data_all.drop(index_to_drop)   # delete

    train_data = data_all[data_all['id'].isin(train_data_name)]
    test_data = data_all[data_all['id'].isin(test_data_name)]
    val_data = data_all[data_all['id'].isin(val_data_name)]

    # data load
    X_train, X_valid, X_test, SA_train, SA_val, SA_test, entropy, entropy_sum, sort, train_mean, train_std = \
        load_data2(train_data, val_data, test_data)

    SA_train1 = np.argmax(SA_train, axis=1)
    SA_train1 = tf.expand_dims(SA_train1, 1)
    sort = tf.cast(sort, tf.float32)
    SA_train1 = tf.cast(SA_train1, tf.float32)
    combine_Y = tf.concat([SA_train1, entropy, entropy_sum, sort], axis=-1)
    SA_test1 = tf.argmax(SA_test, axis=1)
    input_shape = X_train.shape[1:]
    model = base_model[basemodel](input_shape)
    model.summary()
    if basemodel == 'sknet':
        metrics = Metrics_con_sknet(validation=(X_valid, SA_val), acc_threshold=acc_threshold,
                                    model_save_path=model_path)
    else:
        metrics = Metrics_con_resnet(validation=(X_valid, SA_val), acc_threshold=acc_threshold,
                                     model_save_path=model_path)
    for layer in model.layers:
        layer.trainable = True

    SA = np.array([-1 / 8] * 32 + [1 / 8] * 32).reshape(64, 1)  # SA 0 1  -> -1 1
    NSA = -SA
    SA = tf.cast(SA, tf.float32)
    NSA = tf.cast(NSA, tf.float32)
    my_loss = sup_obs(c1=SA, c0=NSA,e1=e1, e0=e0, a=a, )
    losses = {  "projection": my_loss ,  "classifier": "binary_crossentropy", }
    loss_weights = {"projection": 1.0, "classifier": 1.0}
    model.compile(  loss=losses, loss_weights=loss_weights,  # 0.0001
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999),
                    metrics={"projection": None, "classifier": "accuracy"})
    # Data Augmentation
    X_trains = smoothed_filter(X_train)
    X_train = tf.concat([X_train, X_trains], axis=0)
    SA_train = tf.concat([SA_train, SA_train], axis=0)
    combine_Y = tf.concat([combine_Y, combine_Y], axis=0)

    early_stopping = EarlyStoppingByLossVal(monitor='val_classifier_loss', patience=30, restore_best_weights=True)
    history = model.fit(x=X_train, y={"projection": combine_Y, "classifier": SA_train},
                           epochs=epoch, verbose=2,
                           validation_data=(X_valid, { "classifier": SA_val}),
                           callbacks=[metrics, early_stopping],
                           batch_size=batch_size,
                           shuffle=True)

    best_model = find_best_model2(result, k)
    print(f"The best model path is: {best_model}")
    if best_model is not None:
        if basemodel == 'sknet':
            model = load_model(best_model, custom_objects={'Axpby': Axpby, 'loss_fn': my_loss})
        else:
            model.load_weights(best_model)

    score = model.evaluate(X_test, {"classifier": SA_test} , batch_size=512)  # model.evaluate return loss  acc
    loss = round(score[0], 3)
    test_predict_raw = model.predict(X_test)[1]
    test_targ = SA_test1
    test_predict = np.argmax(test_predict_raw, axis=1)
    savefigure(history, result, test_targ, test_predict, test_predict_raw[:, 1], k)  # save picture
    fpr, tpr, thresholds = roc_curve(test_targ, test_predict_raw[:, 1])

    Auc = auc(fpr, tpr)
    ta = confusion_matrix(test_targ, test_predict)
    Acc, Sen, Spe, Pre, F1 = calculate(ta)

    with open(summary, "a") as f:
        f.write('acc-{:.3f}-sen-{:.3f}-spe-{:.3f}-pre-{:.3f}-f1-{:.3f}-auc{:.3f}-loss{:.3f}'.format(Acc, Sen, Spe, Pre, F1, Auc, loss) + '\n')

    ahi = []       # true
    pre_ahi = []   # pred
    for i in  test_data_name:    # per-recording detection
        test_data = data_all[data_all['id'].isin([i])]
        test_PPI = test_data['RRdata']
        test_PPI = list(map(eval, test_PPI))
        test_PPI = np.array(test_PPI[:])
        label = LabelEncoder().fit(test_data.label_SA)
        label_sa = label.transform(test_data.label_SA)

        test_PPI = (test_PPI - train_mean) / train_std
        test_PPI = test_PPI.reshape(test_PPI.shape[0], 500, 1)
        test_PPI = test_PPI.astype('float32')

        test_predict = model.predict(test_PPI)[1]
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
    summary = result + '/summary.txt'  # Record the test results
    if not os.path.exists(result):
        os.makedirs(result)
    data_path = '.\data\label_data_5min_obs.csv'
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
    a = args.a
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu)
    k = 0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
    for train_val_index, test_index in kfold.split(data_name, data_label):
        train_val_data, test_data_name = data_name[train_val_index], data_name[test_index]
        train_val_label, test_label = data_label[train_val_index], data_label[test_index]

        train_val_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
        for train_index, val_index in train_val_kfold.split(train_val_data, train_val_label):
            train_data_name, val_data_name = train_val_data[train_index], train_val_data[val_index]
            train_label, val_label = train_val_label[train_index], train_val_label[val_index]

            model_path = result + '/fold_' + str(k)
            Acc[0, k], Sen[0, k], Spe[0, k], Pre[0, k], F1[0, k], acc[0, k], sen[0, k], spe[0, k], pre[0, k], f1[0, k], mae[0, k], pcc[0, k], Auc[0, k] = \
                my_main(train_data_name, val_data_name, test_data_name, data_path, result, model_path, batch_size, epoch,
                        learning_rate, acc_threshold,  k,  summary, basemodel, a,)
            k = k + 1
            break  # Break out of the loop with a single division.
    with open(summary, "a") as f:
        f.write("segment detection mean acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} Auc:{:.3f}".format(
            np.mean(Acc), np.mean(Sen), np.mean(Spe), np.mean(Pre), np.mean(F1), np.mean(Auc)) + '\n')
        f.write("segment detection std  acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} Auc:{:.3f}".format(
            np.std(Acc), np.std(Sen), np.std(Spe), np.std(Pre), np.std(F1), np.std(Auc)) + '\n')

        f.write("individual detection mean acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} MAE:{:.4f} PCC:{:.4f}".format(
            np.mean(acc), np.mean(sen), np.mean(spe), np.mean(pre), np.mean(f1), np.mean(mae) , np.mean(pcc))  + '\n')
        f.write("individual detection std  acc:{:.3f}   sen{:.3f}  spe:{:.3f}  pre:{:.3f} f1:{:.3f} MAE:{:.4f} PCC:{:.4f}".format(
            np.std(acc), np.std(sen), np.std(spe), np.std(pre), np.std(f1), np.std(mae) , np.std(pcc))  + '\n')
