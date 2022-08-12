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

def BS_BSS(ytrue, ypred): 

    ytrue_rev    = ytrue.copy()
    indices_one  = ytrue_rev == 1
    indices_zero = ytrue_rev == 0
    ytrue_rev[indices_one] = 0 # replacing 1s with 0s
    ytrue_rev[indices_zero] = 1 # replacing 0s with 1s
    
    
    P_c = np.mean(ytrue_rev)
    bs_init = 0
    bss_init = 0
    for e in range(len(ytrue_rev)):
        bss_init  = bs_init + (P_c - ytrue_rev[e])**2 # average value of fog accurence 
        
        if ytrue_rev[e] == 0:
            prob = ypred[e, 1]
            bs_init  = bs_init + (prob - 0)**2
            
        elif ytrue_rev[e] == 1:
            prob = ypred[e, 0]
            bs_init  = bs_init + (prob - 1)**2
            
    BS     = bs_init/len(ytrue_rev)
    BS_ref = bss_init/len(ytrue_rev)
    BSS    = (1-BS)/BS_ref 
    
    return BS, BSS

def skilled_metrics(ytrue, ypred, metric=None): 
    
    # calculate the results based on the range of threshold: 
    threshold_list, result_list = Optimal_Thr(ytrue, ypred)
    
    # select the best result by maximizing them based on PSS
    if metric == 'PSS':
        PSS  = result_list[:, 8]
        raws = np.where(PSS == np.amax(PSS))[-1] 
        length_raws = len(raws) 
        if length_raws == 1:
            accuray_list      = result_list[raws[0], :] 
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list      = result_list[raws[-1], :] 
            optimal_threshold = threshold_list[raws[-1]]

    
    # select the best result by maximizing them based on HSS
    elif metric == 'HSS':
        
        HSS  = result_list[:, 9]
        raws = np.where(HSS == np.amax(HSS))[-1] 
        length_raws = len(raws) 
        if length_raws == 1:
            accuray_list      = result_list[raws[0], :] 
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list      = result_list[raws[-1], :]
            optimal_threshold = threshold_list[raws[-1]]
            
    # select the best result by maximizing them based on CSS
    elif metric == 'CSS':
        CSS  = result_list[:, 11]
        CSS = [x for x in CSS if np.isnan(x) == False]
        raws = np.where(CSS == np.amax(CSS))[-1] 
        length_raws = len(raws) 
        if length_raws == 1:
            accuray_list = result_list[raws[0], :] 
            optimal_threshold = threshold_list[raws[0]]
        else:
            accuray_list = result_list[raws[-1], :]
            optimal_threshold = threshold_list[raws[-1]]
            
    if accuray_list[4] == 1.0:
        accuray_list[4] = 0.999  
        
    if accuray_list[5] <= 0:
        accuray_list[5] = 0.009 
    
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
    

    if POD == 1.0:
        POD = 0.999  
        
    if F <= 0:
        F = 0.009 
        
    SEDI = (log(F) - log(POD) - log(1-F) + log(1-POD))/(log(F) + log(POD) + log(1-F) + log(1-POD))
    
    
    output = [Hit, MISS, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS, SEDI]
    #output = [Hit, MISS, FA, CR, POD, F, FAR, CSI, PSS, HSS, ORSS, CSS]
    
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
        
    O_POD = accuray_list[4]
    O_HSS = accuray_list[9]
    O_FAR = accuray_list[6] 
    POD = results[:, 4] 
    FAR = results[:, 6]
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
    
    
    
    
def print_results(tr_acc, te_acc):
    print("Hit/Miss cases: ")
    print(f"       Hit    Miss    False Alarm   Correct rejected")
    print(f"Train  {int(tr_acc[0])}      {int(tr_acc[1])}         {int(tr_acc[2])}              {int(tr_acc[3])}")
    print(f"Test   {te_acc[0]}      {te_acc[1]}        {te_acc[2]}              {te_acc[3]}")
    print()
    print("Metrics:")
    print(f"       POD     F    FAR    CSI    PSS     HSS   ORSS   CSS   SEDI")
    print(f"Train  {tr_acc[4]:.2f}   {tr_acc[4]:.2f}  {tr_acc[6]:.2f}   {tr_acc[7]:.2f}   {tr_acc[8]:.2f}   {tr_acc[9]:.2f}   {tr_acc[10]:.2f}   {tr_acc[11]:.2f}  {tr_acc[12]:.2f}")
    print(f"Test   {te_acc[4]:.2f}   {te_acc[5]:.2f}  {te_acc[6]:.2f}   {te_acc[7]:.2f}   {te_acc[8]:.2f}   {te_acc[9]:.2f}   {te_acc[10]:.2f}   {te_acc[11]:.2f}  {te_acc[12]:.2f}")   
    
    
DEFAULT_NUM_BINS = 20
RELIABILITY_LINE_COLOUR = np.array([228, 26, 28], dtype=float) / 255
RELIABILITY_LINE_WIDTH = 3
PERFECT_LINE_COLOUR = np.full(3, 152. / 255)
PERFECT_LINE_WIDTH = 2

NO_SKILL_LINE_COLOUR = np.array([31, 120, 180], dtype=float) / 255
NO_SKILL_LINE_WIDTH = 2
SKILL_AREA_TRANSPARENCY = 0.2
CLIMATOLOGY_LINE_COLOUR = np.full(3, 152. / 255)
CLIMATOLOGY_LINE_WIDTH = 2

HISTOGRAM_FACE_COLOUR = np.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = np.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 2

HISTOGRAM_LEFT_EDGE_COORD = 0.575
HISTOGRAM_BOTTOM_EDGE_COORD = 0.175
HISTOGRAM_WIDTH = 0.3
HISTOGRAM_HEIGHT = 0.3

HIST_LEFT_EDGE_FOR_REGRESSION = 0.575
HIST_WIDTH_FOR_REGRESSION = 0.3
HIST_BOTTOM_EDGE_FOR_REGRESSION = 0.225
HIST_HEIGHT_FOR_REGRESSION = 0.25

HISTOGRAM_X_TICK_VALUES = np.linspace(0, 1, num=6, dtype=float)
HISTOGRAM_X_TICKS_FOR_REGRESSION = np.linspace(0, 0.02, num=11)
HISTOGRAM_Y_TICK_SPACING = 0.1

FIGURE_WIDTH_INCHES = 10
FIGURE_HEIGHT_INCHES = 10

FONT_SIZE = 20
plt.rc('font', size=FONT_SIZE)
plt.rc('axes', titlesize=FONT_SIZE)
plt.rc('axes', labelsize=FONT_SIZE)
plt.rc('xtick', labelsize=FONT_SIZE)
plt.rc('ytick', labelsize=FONT_SIZE)
plt.rc('legend', fontsize=FONT_SIZE)
plt.rc('figure', titlesize=FONT_SIZE)

def _get_histogram(input_values, num_bins, min_value, max_value):
    """Creates histogram with uniform bin-spacing.
    E = number of input values
    B = number of bins
    :param input_values: length-E numpy array of values to bin.
    :param num_bins: Number of bins (B).
    :param min_value: Minimum value.  Any input value < `min_value` will be
        assigned to the first bin.
    :param max_value: Max value.  Any input value > `max_value` will be
        assigned to the last bin.
    :return: inputs_to_bins: length-E numpy array of bin indices (integers).
    """

    bin_cutoffs = np.linspace(min_value, max_value, num=num_bins + 1)

    inputs_to_bins = np.digitize(
        input_values, bin_cutoffs, right=False
    ) - 1

    inputs_to_bins[inputs_to_bins < 0] = 0
    inputs_to_bins[inputs_to_bins > num_bins - 1] = num_bins - 1

    return inputs_to_bins


def _vertices_to_polygon_object(x_vertices, y_vertices):
    """Converts two arrays of vertices to `shapely.geometry.Polygon` object.
    V = number of vertices
    This method allows for simple polygons only (no disjoint polygons, no
    holes).
    :param x_vertices: length-V numpy array of x-coordinates.
    :param y_vertices: length-V numpy array of y-coordinates.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    list_of_vertices = []

    for i in range(len(x_vertices)):
        list_of_vertices.append(
            (x_vertices[i], y_vertices[i])
        )

    return shapely.geometry.Polygon(shell=list_of_vertices)


def _plot_background(axes_object, observed_labels):
    """Plots background of attributes diagram.
    E = number of examples
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    """

    # Plot positive-skill area.
    climatology = np.mean(observed_labels.astype(float))
    skill_area_colour = matplotlib.colors.to_rgba(
        NO_SKILL_LINE_COLOUR, SKILL_AREA_TRANSPARENCY)

    x_vertices_left = np.array([0, climatology, climatology, 0, 0])
    y_vertices_left = np.array([0, 0, climatology, climatology / 2, 0])

    left_polygon_object = _vertices_to_polygon_object(
        x_vertices=x_vertices_left, y_vertices=y_vertices_left)
    left_polygon_patch = PolygonPatch(
        left_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour)
    axes_object.add_patch(left_polygon_patch)

    x_vertices_right = np.array(
        [climatology, 1, 1, climatology, climatology]
    )
    y_vertices_right = np.array(
        [climatology, (1 + climatology) / 2, 1, 1, climatology]
    )

    right_polygon_object = _vertices_to_polygon_object(
        x_vertices=x_vertices_right, y_vertices=y_vertices_right)
    right_polygon_patch = PolygonPatch(
        right_polygon_object, lw=0, ec=skill_area_colour, fc=skill_area_colour)
    axes_object.add_patch(right_polygon_patch)

    # Plot no-skill line (at edge of positive-skill area).
    no_skill_x_coords = np.array([0, 1], dtype=float)
    no_skill_y_coords = np.array([climatology, 1 + climatology]) / 2
    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords, color=NO_SKILL_LINE_COLOUR,
        linestyle='solid', linewidth=NO_SKILL_LINE_WIDTH)

    # Plot climatology line (vertical).
    climo_line_x_coords = np.full(2, climatology)
    climo_line_y_coords = np.array([0, 1], dtype=float)
    axes_object.plot(
        climo_line_x_coords, climo_line_y_coords, color=CLIMATOLOGY_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMATOLOGY_LINE_WIDTH)

    # Plot no-resolution line (horizontal).
    no_resolution_x_coords = climo_line_y_coords + 0.
    no_resolution_y_coords = climo_line_x_coords + 0.
    axes_object.plot(
        no_resolution_x_coords, no_resolution_y_coords,
        color=CLIMATOLOGY_LINE_COLOUR, linestyle='dashed',
        linewidth=CLIMATOLOGY_LINE_WIDTH)


def _floor_to_nearest(input_value_or_array, increment):
    """Rounds number(s) down to the nearest multiple of `increment`.
    :param input_value_or_array: Input (either scalar or numpy array).
    :param increment: Increment (or rounding base -- whatever you want to call
        it).
    :return: output_value_or_array: Rounded version of `input_value_or_array`.
    """

    return increment * np.floor(input_value_or_array / increment)


def _plot_forecast_histogram(figure_object, num_examples_by_bin):
    """Plots forecast histogram as inset in the attributes diagram.
    B = number of bins
    :param figure_object: Instance of `matplotlib.figure.Figure`.  Will plot in
        this figure.
    :param num_examples_by_bin: length-B numpy array, where
        num_examples_by_bin[j] = number of examples in [j]th forecast bin.
    """

    num_bins = len(num_examples_by_bin)
    bin_frequencies = (
        num_examples_by_bin.astype(float) / np.sum(num_examples_by_bin)
    )

    forecast_bin_edges = np.linspace(0, 1, num=num_bins + 1, dtype=float)
    forecast_bin_width = forecast_bin_edges[1] - forecast_bin_edges[0]
    forecast_bin_centers = forecast_bin_edges[:-1] + forecast_bin_width / 2

    inset_axes_object = figure_object.add_axes(
        [HISTOGRAM_LEFT_EDGE_COORD, HISTOGRAM_BOTTOM_EDGE_COORD,
         HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT]
    )

    inset_axes_object.bar(
        forecast_bin_centers, bin_frequencies, forecast_bin_width,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH)

    max_y_tick_value = _floor_to_nearest(
        1.05 * np.max(bin_frequencies), HISTOGRAM_Y_TICK_SPACING)
    num_y_ticks = 1 + int(np.round(
        max_y_tick_value / HISTOGRAM_Y_TICK_SPACING
    ))

    y_tick_values = np.linspace(0, max_y_tick_value, num=num_y_ticks)
    plt.yticks(y_tick_values, axes=inset_axes_object)
    plt.xticks(HISTOGRAM_X_TICK_VALUES, axes=inset_axes_object)

    inset_axes_object.set_xlim(0, 1)
    inset_axes_object.set_ylim(0, 1.05 * np.max(bin_frequencies))


def _plot_forecast_hist_for_regression(
        figure_object, mean_forecast_by_bin, num_examples_by_bin):
    """Plots forecast histogram for regression.
    B = number of bins
    :param figure_object: Will plot histogram as inset in this figure (instance
        of `matplotlib.figure.Figure`).
    :param mean_forecast_by_bin: length-B numpy array of mean forecast values.
    :param num_examples_by_bin: length-B numpy array of example counts.
    """

    bin_frequencies = (
        num_examples_by_bin.astype(float) / np.sum(num_examples_by_bin)
    )

    num_bins = len(num_examples_by_bin)
    forecast_bin_width = (
        (np.max(mean_forecast_by_bin) - np.min(mean_forecast_by_bin)) /
        (num_bins - 1)
    )

    inset_axes_object = figure_object.add_axes([
        HIST_LEFT_EDGE_FOR_REGRESSION, HIST_BOTTOM_EDGE_FOR_REGRESSION,
        HIST_WIDTH_FOR_REGRESSION, HIST_HEIGHT_FOR_REGRESSION
    ])

    inset_axes_object.bar(
        mean_forecast_by_bin, bin_frequencies, forecast_bin_width,
        color=HISTOGRAM_FACE_COLOUR, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH)

    max_y_tick_value = _floor_to_nearest(
        1.05 * np.max(bin_frequencies), HISTOGRAM_Y_TICK_SPACING
    )
    num_y_ticks = 1 + int(np.round(
        max_y_tick_value / HISTOGRAM_Y_TICK_SPACING
    ))

    y_tick_values = np.linspace(0, max_y_tick_value, num=num_y_ticks)
    plt.yticks(y_tick_values, axes=inset_axes_object)
    plt.xticks(HISTOGRAM_X_TICKS_FOR_REGRESSION, axes=inset_axes_object,
                  rotation=90.)

    inset_axes_object.set_xlim(
        0, np.max(mean_forecast_by_bin) + forecast_bin_width
    )
    inset_axes_object.set_ylim(0, 1.05 * np.max(bin_frequencies))


def _get_points_in_regression_relia_curve(observed_values, forecast_values,
                                          num_bins):
    """Creates points for regression-based reliability curve.
    E = number of examples
    B = number of bins
    :param observed_values: length-E numpy array of observed target values.
    :param forecast_values: length-E numpy array of forecast target values.
    :param num_bins: Number of bins for forecast value.
    :return: mean_forecast_by_bin: length-B numpy array of mean forecast values.
    :return: mean_observation_by_bin: length-B numpy array of mean observed
        values.
    :return: num_examples_by_bin: length-B numpy array with number of examples
        in each forecast bin.
    """

    inputs_to_bins = _get_histogram(
        input_values=forecast_values, num_bins=num_bins,
        min_value=np.min(forecast_values),
        max_value=np.max(forecast_values)
    )

    mean_forecast_by_bin = np.full(num_bins, np.nan)
    mean_observation_by_bin = np.full(num_bins, np.nan)
    num_examples_by_bin = np.full(num_bins, -1, dtype=int)

    for k in range(num_bins):
        these_example_indices = np.where(inputs_to_bins == k)[0]
        num_examples_by_bin[k] = len(these_example_indices)

        mean_forecast_by_bin[k] = np.mean(
            forecast_values[these_example_indices]
        )

        mean_observation_by_bin[k] = np.mean(
            observed_values[these_example_indices]
        )

    return mean_forecast_by_bin, mean_observation_by_bin, num_examples_by_bin


def get_points_in_relia_curve(
        observed_labels, forecast_probabilities, num_bins):
    """Creates points for reliability curve.
    The reliability curve is the main component of the attributes diagram.
    E = number of examples
    B = number of bins
    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins for forecast probability.
    :return: mean_forecast_probs: length-B numpy array of mean forecast
        probabilities.
    :return: mean_event_frequencies: length-B numpy array of conditional mean
        event frequencies.  mean_event_frequencies[j] = frequency of label 1
        when forecast probability is in the [j]th bin.
    :return: num_examples_by_bin: length-B numpy array with number of examples
        in each forecast bin.
    """

    assert np.all(np.logical_or(
        observed_labels == 0, observed_labels == 1
    ))

    assert np.all(np.logical_and(
        forecast_probabilities >= 0, forecast_probabilities <= 1
    ))

    assert num_bins > 1

    inputs_to_bins = _get_histogram(
        input_values=forecast_probabilities, num_bins=num_bins, min_value=0.,
        max_value=1.)

    mean_forecast_probs = np.full(num_bins, np.nan)
    mean_event_frequencies = np.full(num_bins, np.nan)
    num_examples_by_bin = np.full(num_bins, -1, dtype=int)

    for k in range(num_bins):
        these_example_indices = np.where(inputs_to_bins == k)[0]
        num_examples_by_bin[k] = len(these_example_indices)

        mean_forecast_probs[k] = np.mean(
            forecast_probabilities[these_example_indices])

        mean_event_frequencies[k] = np.mean(
            observed_labels[these_example_indices].astype(float)
        )

    return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin


def plot_reliability_curve(
        observed_labels, forecast_probabilities, num_bins=DEFAULT_NUM_BINS,
        axes_object=None):
    """Plots reliability curve.
    E = number of examples
    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :param num_bins: Number of bins for forecast probability.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).  If `axes_object is None`,
        will create new axes.
    :return: mean_forecast_probs: See doc for `get_points_in_relia_curve`.
    :return: mean_event_frequencies: Same.
    :return: num_examples_by_bin: Same.
    """

    mean_forecast_probs, mean_event_frequencies, num_examples_by_bin = (
        get_points_in_relia_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities, num_bins=num_bins)
    )

    if axes_object is None:
        _, axes_object = plt.subplots(
            1, 1, figsize=(10, 10)
        )

    perfect_x_coords = np.array([0, 1], dtype=float)
    perfect_y_coords = perfect_x_coords + 0.
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=PERFECT_LINE_COLOUR,
        linestyle='dashed', linewidth=PERFECT_LINE_WIDTH)

    real_indices = np.where(np.invert(np.logical_or(
        np.isnan(mean_forecast_probs), np.isnan(mean_event_frequencies)
    )))[0]

    axes_object.plot(
        mean_forecast_probs[real_indices], mean_event_frequencies[real_indices],
        color=RELIABILITY_LINE_COLOUR,
        linestyle='solid', linewidth=RELIABILITY_LINE_WIDTH)

    axes_object.set_xlabel('Forecast probability')
    axes_object.set_ylabel('Conditional event frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin


def plot_regression_relia_curve(
        observed_values, forecast_values, num_bins=DEFAULT_NUM_BINS,
        figure_object=None, axes_object=None):
    """Plots reliability curve for regression.
    :param observed_values: See doc for `get_points_in_regression_relia_curve`.
    :param forecast_values: Same.
    :param num_bins: Same.
    :param figure_object: See doc for `plot_attributes_diagram`.
    :param axes_object: Same.
    """

    mean_forecast_by_bin, mean_observation_by_bin, num_examples_by_bin = (
        _get_points_in_regression_relia_curve(
            observed_values=observed_values, forecast_values=forecast_values,
            num_bins=num_bins)
    )

    if figure_object is None or axes_object is None:
        figure_object, axes_object = plt.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    _plot_forecast_hist_for_regression(
        figure_object=figure_object, mean_forecast_by_bin=mean_forecast_by_bin,
        num_examples_by_bin=num_examples_by_bin)

    max_forecast_or_observed = max([
        np.max(forecast_values), np.max(observed_values)
    ])

    perfect_x_coords = np.array([0., max_forecast_or_observed])
    perfect_y_coords = perfect_x_coords + 0.
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=PERFECT_LINE_COLOUR,
        linestyle='dashed', linewidth=PERFECT_LINE_WIDTH)

    real_indices = np.where(np.invert(np.logical_or(
        np.isnan(mean_forecast_by_bin), np.isnan(mean_observation_by_bin)
    )))[0]

    axes_object.plot(
        mean_forecast_by_bin[real_indices],
        mean_observation_by_bin[real_indices],
        color=RELIABILITY_LINE_COLOUR,
        linestyle='solid', linewidth=RELIABILITY_LINE_WIDTH)

    axes_object.set_xlabel('Forecast value')
    axes_object.set_ylabel('Conditional mean observation')
    axes_object.set_xlim(0., max_forecast_or_observed)
    axes_object.set_ylim(0., max_forecast_or_observed)


def plot_attributes_diagram(
        observed_labels, forecast_probabilities, num_bins=DEFAULT_NUM_BINS,
        figure_object=None, axes_object=None):
    """Plots attributes diagram.
    :param observed_labels: See doc for `plot_reliability_curve`.
    :param forecast_probabilities: Same.
    :param num_bins: Same.
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).  If `figure_object is None`, will create a
        new one.
    :param axes_object: See doc for `plot_reliability_curve`.
    :return: mean_forecast_probs: See doc for `get_points_in_relia_curve`.
    :return: mean_event_frequencies: Same.
    :return: num_examples_by_bin: Same.
    """

    mean_forecast_probs, mean_event_frequencies, num_examples_by_bin = (
        get_points_in_relia_curve(
            observed_labels=observed_labels,
            forecast_probabilities=forecast_probabilities, num_bins=num_bins)
    )

    if figure_object is None or axes_object is None:
        figure_object, axes_object = plt.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    _plot_background(axes_object=axes_object, observed_labels=observed_labels)
    _plot_forecast_histogram(figure_object=figure_object,
                             num_examples_by_bin=num_examples_by_bin)

    plot_reliability_curve(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities, num_bins=num_bins,
        axes_object=axes_object)

    return mean_forecast_probs, mean_event_frequencies, num_examples_by_bin
    
    
    
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