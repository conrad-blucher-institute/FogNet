import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def Optimal_Thr(y, ypred, save_name):

    length = len(ypred)
    ypred_ = [0]*length
    results = numpy.empty(shape = (900, 8), dtype='float')
    th_list = numpy.arange(0.1, 1, 0.001).tolist()

    for j in range(900):
        th = th_list[j]
        for i in range(length):
            prob = ypred[i, 1]
            if prob > th:
                ypred_[i] = 1
            else:
                ypred_[i] = 0
        ypred_ = numpy.array(ypred_)
        tn, fp, fn, tp = confusion_matrix(y, ypred_).ravel()
        a = tn     # Hit
        #print('Hit: ', a,  file=open(save_name, "a"))
        b = fn      # false alarm
        #print('False alarm: ', b,  file=open(save_name, "a"))
        c = fp      # miss
        #print('Miss: ', c,  file=open(save_name, "a"))
        d = tp    # correct rejection
        #print('Correct rejection: ', d,  file=open(save_name, "a"))

        POD = a/(a+c)
        results[j, 0] = POD

        F   = b/(b+d)
        results[j, 1] = F

        FAR  = b/(a+b)
        results[j,2] = FAR

        CSI = a/(a+b+c)
        results[j, 3] = CSI

        PSS = ((a*d)-(b*c))/((b+d)*(a+c))
        results[j, 4] = PSS

        HSS = (2*((a*d)-(b*c)))/(((a+c)*(c+d))+((a+b)*(b+d)))
        results[j, 5] = HSS

        ORSS = ((a*d)-(b*c))/((a*d)+(b*c))
        results[j, 6] = ORSS

        CSS = ((a*d)-(b*c))/((a+b)*(c+d))
        results[j, 7] = CSS


    return th_list, results



def skilled_metrics(y, ypred, metric, save_name):
    threshold_list, result_list = Optimal_Thr(y, ypred, save_name)
    if metric == 'PSS':
        PSS = result_list[:, 4]
        raws = numpy.where(PSS == numpy.amax(PSS))[-1]
        length_raws = len(raws)
        if length_raws == 1:
            accuray_list = result_list[raws[0], :]
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list = result_list[raws[-1], :]
            optimal_threshold = threshold_list[raws[-1]]

    if metric == 'HSS':

        HSS = result_list[:, 5]
        raws = numpy.where(HSS == numpy.amax(HSS))[-1]
        length_raws = len(raws)
        if length_raws == 1:
            accuray_list = result_list[raws[0], :]
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list = result_list[raws[-1], :]
            optimal_threshold = threshold_list[raws[-1]]

    print('The optima threshold is: ', optimal_threshold,  file=open(save_name, "a"))
    print('POD  : ', accuray_list[0],  file=open(save_name, "a"))
    print('F    : ', accuray_list[1],  file=open(save_name, "a"))
    print('FAR  : ', accuray_list[2],  file=open(save_name, "a"))
    print('CSI  : ', accuray_list[3],  file=open(save_name, "a"))
    print('PSS  : ', accuray_list[4],  file=open(save_name, "a"))
    print('HSS  : ', accuray_list[5],  file=open(save_name, "a"))
    print('ORSS : ', accuray_list[6],  file=open(save_name, "a"))
    print('CSS  : ', accuray_list[7],  file=open(save_name, "a"))

    return optimal_threshold




def confusion_cnn(y, ypred, threshold, save_name):

    #classnames  = ['0 =< vis =< 1 miles', '1< vis =< 2 miles', '2 < vis =< 4 miles', '4 < vis < 6 miles','6 =< vis < 10 miles']
    # confusion Matrix:
    length = len(ypred)
    ypred_ = [0]*length
    '''for i in range(length):
        ypred_[i] = numpy.argmax(ypred[i, :]) '''
    ypred_ = numpy.array(ypred_)

    for i in range(length):
        prob = ypred[i, 1]
        if prob > threshold:
            ypred_[i] = 1
        else:
            ypred_[i] = 0

    ypred_ = numpy.array(ypred_)

    tn, fp, fn, tp = confusion_matrix(y, ypred_).ravel()
    a = tn     # Hit
    #print('Hit: ', a,  file=open(save_name, "a"))
    b = fn      # false alarm
    #print('False alarm: ', b,  file=open(save_name, "a"))
    c = fp      # miss
    #print('Miss: ', c,  file=open(save_name, "a"))
    d = tp    # correct rejection
    #print('Correct rejection: ', d,  file=open(save_name, "a"))

    POD = a/(a+c)
    print('POD  : ', POD,  file=open(save_name, "a"))
    F   = b/(b+d)
    print('F    : ', F,  file=open(save_name, "a"))

    FAR  = b/(a+b)
    print('FAR  : ', FAR,  file=open(save_name, "a"))

    CSI = a/(a+b+c)
    print('CSI  : ', CSI,  file=open(save_name, "a"))

    PSS = ((a*d)-(b*c))/((b+d)*(a+c))
    print('PSS  : ', PSS,  file=open(save_name, "a"))

    HSS = (2*((a*d)-(b*c)))/(((a+c)*(c+d))+((a+b)*(b+d)))
    print('HSS  : ', HSS,  file=open(save_name, "a"))

    ORSS = ((a*d)-(b*c))/((a*d)+(b*c))
    print('ORSS : ', ORSS,  file=open(save_name, "a"))

    CSS = ((a*d)-(b*c))/((a+b)*(c+d))
    print('CSS  : ', CSS,  file=open(save_name, "a"))






def plot_loss_function(history, save_name):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize = (15, 10))
    epochs = range(1, len(val_acc) + 1)
    plt.plot(epochs, train_acc, 'o-',  label = 'Training Score')
    plt.plot(epochs, val_acc, 'o-',  label = 'Validation Score')


    plt.title('Validation Curve of CNN Model')
    plt.grid()
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.figure(figsize = (15, 10))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(epochs, train_loss, 'o-',  label = 'Training loss')
    plt.plot(epochs, val_loss, 'o-',  label  = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
    plt.savefig(save_name)


def plot_ROC_Curve (n_classes, y, y_pred, save_name):
    n_classes = n_classes
    plt.figure(figsize = (10, 10))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = (['m', 'darkblue', 'coral', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class Visibility')
    plt.legend(loc="lower right", shadow=True, fontsize =  'large')
    plt.show()
    plt.savefig(save_name)
