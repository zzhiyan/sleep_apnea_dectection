from sklearn.metrics import confusion_matrix, roc_curve , auc
import matplotlib.pyplot as plt
from package.uitils import plot_confusion_matrix
def savefigure(history, result, test_targ, test_predict, test_predict_raw, k):
    fpr, tpr, thresholds_keras = roc_curve(test_targ, test_predict_raw)
    ta = confusion_matrix(test_targ, test_predict)
    Auc = auc(fpr, tpr)
    fig1 = plt.figure()
    plt.title("accuracy")
    print(history.history.keys())  # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    plt.plot(history.history['accuracy'], label="train_acc")
    plt.plot(history.history['val_accuracy'], label="val_acc")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    fig1.savefig(result + '/Acc{}.png'.format(k), dpi=600, format='png')

    fig2 = plt.figure()
    plt.title("loss")
    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig2.savefig(result + '/Loss{}.png'.format(k), dpi=600, format='png')

    # Plot non-normalized confusion matrix
    class_names = ["N", "SAS"]
    fig3 = plt.figure(figsize=(3, 4))
    plot_confusion_matrix(ta, title="Confusion Matrix", classes=class_names)
    # plt.show()
    fig3.savefig(result + '/Confusion_Matrix{}.png'.format(k), dpi=600, format='png')

    # Plot ROC
    fig4 = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot(fpr, tpr, label='ROC-flod-1 AUC:{:.4f}'.format(Auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc=4)
    fig4.savefig(result + '/ROC{}.png'.format(k), dpi=600, format='png')