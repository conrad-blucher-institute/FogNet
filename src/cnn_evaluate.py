import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import math
from math import log



def Optimal_Thr(ytrue, ypred): 
    ypred_ = np.zeros_like(ytrue)
    results = np.empty(shape = (1000, 12), dtype='float')
    th_list = np.arange(0, 1, 0.001).tolist()
    for j in range(1000):
        th = th_list[j]
        for i in range(len(ytrue)):
            prob = ypred[i, 1] 
            if prob > th:
                ypred_[i] = 1
            else:
                ypred_[i] = 0    
        Hit, miss, FA, CR = confusion_matrix(ytrue, ypred_).ravel()
        results[j, 0] = int(Hit)
        results[j, 1] = int(miss)
        results[j, 2] = int(FA)
        results[j, 3] = int(CR)
        
        POD = Hit/(Hit+miss)
        results[j, 4] = POD
        
        F   = FA/(FA+CR)
        results[j, 5] = F

        FAR  = FA/(Hit+FA)
        results[j, 6] = FAR

        CSI = Hit/(Hit+FA+miss)
        results[j, 7] = CSI

        PSS = ((Hit*CR)-(FA*miss))/((FA+CR)*(Hit+miss))
        results[j, 8] = PSS

        HSS = (2*((Hit*CR)-(FA*miss)))/(((Hit+miss)*(miss+CR))+((Hit+FA)*(FA+CR)))
        results[j, 9] = HSS

        ORSS = ((Hit*CR)-(FA*miss))/((Hit*CR)+(FA*miss))
        results[j, 10] = ORSS

        CSS = ((Hit*CR)-(FA*miss))/((Hit+FA)*(miss+CR))
        results[j, 11] = CSS
                
        
    return th_list, results


def skilled_metrics(ytrue, ypred, metric=None): 
    
    # calculate the results based on the range of threshold: 
    threshold_list, result_list = Optimal_Thr(ytrue, ypred)
    
    # select the best result by maximizing them based on PSS
    if metric == 'PSS':
        PSS = result_list[:, 8]
        raws = np.where(PSS == np.amax(PSS))[-1] 
        length_raws = len(raws) 
        if length_raws == 1:
            accuray_list = result_list[raws[0], :] 
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list = result_list[raws[-1], :] 
            optimal_threshold = threshold_list[raws[-1]]

    
    # select the best result by maximizing them based on HSS
    elif metric == 'HSS':
        
        HSS = result_list[:, 9]
        raws = np.where(HSS == np.amax(HSS))[-1] 
        length_raws = len(raws) 
        if length_raws == 1:
            accuray_list = result_list[raws[0], :] 
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list = result_list[raws[-1], :]
            optimal_threshold = threshold_list[raws[-1]]
            
    # select the best result by maximizing them based on CSS
    elif metric == 'CSS':
        
        CSS = result_list[:, 11]
        raws = np.where(CSS == np.amax(CSS))[-1] 
        length_raws = len(raws) 
        if length_raws == 1:
            accuray_list = result_list[raws[0], :] 
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list = result_list[raws[-1], :]
            optimal_threshold = threshold_list[raws[-1]]
            
    
    
    SEDI = (log(accuray_list[5]) - log(accuray_list[4]) - log(1-accuray_list[5]) + log(1-accuray_list[4]))/(log(accuray_list[5]) + log(accuray_list[4]) + log(1-accuray_list[5]) + log(1-accuray_list[4]))
    accuray_list = np.append(accuray_list, SEDI)
    return optimal_threshold, accuray_list, result_list


def test_eval(ytrue, ypred, threshold = None): 
    
    length = len(ypred) 
    ypred_ = [0]*length

    for i in range(length):
        prob = ypred[i, 1] 
        if prob > threshold:
            ypred_[i] = 1
        else:
            ypred_[i] = 0
            
            
    ypred_ = np.array(ypred_)

    Hit, MISS, FA, CR = confusion_matrix(ytrue, ypred_).ravel()
    
    POD  = Hit/(Hit+MISS)
    F    = FA/(FA+CR)
    FAR  = FA/(Hit+FA)
    CSI  = Hit/(Hit+FA+MISS)
    PSS  = ((Hit*CR)-(FA*MISS))/((FA+CR)*(Hit+MISS))
    HSS  = (2*((Hit*CR)-(FA*MISS)))/(((Hit+MISS)*(MISS+CR))+((Hit+FA)*(FA+CR)))
    ORSS = ((Hit*CR)-(FA*MISS))/((Hit*CR)+(FA*MISS))
    CSS  = ((Hit*CR)-(FA*MISS))/((Hit+FA)*(MISS+CR))
    SEDI = (log(F) - log(POD) - log(1-F) + log(1-POD))/(log(F) + log(POD) + log(1-F) + log(1-POD))
    
    
    output = [Hit, MISS, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS, SEDI]
    
    return output





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

    
def ROC_Curve_Plot(optimal_threshold, accuray_list, results): 
    
    plt.figure(figsize = (10, 10))

    axes_object = None
    if axes_object is None:
        _, axes_object = plt.subplots(
            1, 1, figsize=(10, 10)
        )
        
    O_POD = accuray_list[0]
    O_HSS = accuray_list[5]
    O_FAR = accuray_list[2] 
    POD = results[:, 0] 
    FAR = results[:, 2]
    FAR = np.nan_to_num(FAR, nan = 1)
    FAR = np.sort(FAR)


    Roc_Auc = auc(FAR, POD)     
        
        
    #textstr = 'Max HSS=%.2f' % (O_HSS) 
    textstr = '\n'.join((
    r'Optimal Threshold=%.2f' % (optimal_threshold, ),
    r'Max HSS=%.2f' % (O_HSS, ), 
    r'FAR=%.2f' % (O_FAR, ), 
    r'POD=%.2f' % (O_POD, )))
    
    props = dict(boxstyle='round', facecolor='cyan', alpha=0.1)
    
    plt.plot(FAR, POD, linewidth=3, color = 'red')
    plt.plot(O_FAR, O_POD, 'bo', markersize=10) 
    plt.plot([0, 1], [0, 1], 'k--')
    
    axes_object.text(O_FAR + 0.02, O_POD - 0.02, textstr, size=50, rotation=0, transform=axes_object.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    plt.xlim([-0.005, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('FAR (probability of false detection)',  fontsize=20)
    plt.ylabel('POD (probability of detection)', fontsize=20)
    title_string = 'ROC curve (AUC = {0:.3f})'.format(Roc_Auc)
    plt.title(title_string, fontsize=20)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''def Optimal_Thr(y, ypred, save_name):

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
    
    
    def confusion_cnn(y, ypred, threshold, save_name):

    #classnames  = ['0 =< vis =< 1 miles', '1< vis =< 2 miles', '2 < vis =< 4 miles', '4 < vis < 6 miles','6 =< vis < 10 miles']
    # confusion Matrix:
    length = len(ypred)
    ypred_ = [0]*length

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

    '''


'''def skilled_metrics(y, ypred, metric, save_name):
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

    return optimal_threshold'''