from sklearn.metrics import confusion_matrix, roc_curve , auc
import matplotlib.pyplot as plt
from package.uitils import plot_confusion_matrix
def savefigure(history, result, test_targ, test_predict, test_predict_raw, k):
    fpr, tpr, thresholds = roc_curve(test_targ, test_predict_raw)
    ta = confusion_matrix(test_targ, test_predict)
    Auc = auc(fpr, tpr)
    fig1 = plt.figure(1)
    plt.title("accuracy")
    # print(history.history.keys())
    # ['loss', 'projection_loss', 'classifier_loss', 'classifier_accuracy', 'val_loss', 'val_projection_loss', 'val_classifier_loss', 'val_classifier_accuracy']

    plt.plot(history.history['classifier_accuracy'], label="classifier_train_acc_flod_{}".format(k) )
    plt.plot(history.history['val_classifier_accuracy'], label="classifier_val_acc_flod_{}".format(k) )
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    fig1.savefig(result + '/acc{}.png'.format(k), dpi=600, format='png')

    fig2 = plt.figure(2)
    plt.title("Model loss")
    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['val_classifier_loss'], label="val_classifier_loss")
    plt.plot(history.history['classifier_loss'], label="train_classifier_loss")
    plt.plot(history.history['projection_loss'], label="train_projection_loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    fig2.savefig(result + '/loss{}.png'.format(k), dpi=600, format='png')

    # Plot non-normalized confusion matrix
    class_names = ["N", "SAS"]
    fig3 = plt.figure(figsize=(3, 4))
    plot_confusion_matrix(ta, title="Confusion Matrix", classes=class_names)
    # plt.show()
    fig3.savefig(result + '/Confusion_Matrix{}.png'.format(k), dpi=600, format='png')

    # Plot ROC
    fig4 = plt.figure(4)
    plt.plot([0, 1], [0, 1], 'k--', )
    plt.xlim([0.00, 1.00])
    plt.ylim([0.00, 1.00])
    plt.plot(fpr, tpr, label='ROC flod {} (AUC:{:.2f})'.format(k, Auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    # plt.show()
    fig4.savefig(result + '/ROC.png', dpi=600, format='png')