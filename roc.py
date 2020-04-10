import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pickle, os

def calculate_metrics(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)
    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else float(true_positives) / float(true_positives + false_negatives)
    false_positive_rate = 0 if (false_positives + true_negatives == 0) else float(false_positives) / float(false_positives + true_negatives)
    return true_positive_rate, false_positive_rate

def calculate_roc_values(thresholds, distances, labels):
    num_thresholds = len(thresholds)
    true_positive_rate = np.zeros((num_thresholds))
    false_positive_rate = np.zeros((num_thresholds))
    for threshold_index, threshold in enumerate(thresholds):
        true_positive_rate[threshold_index], false_positive_rate[threshold_index] = calculate_metrics(
            threshold=threshold, dist=distances, actual_issame=labels
        )
    return true_positive_rate, false_positive_rate

def eval(distances, labels):
    """Entry Point
    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        roc_auc: Area Under the Receiver operating characteristic (ROC) metric.
    """
    # Calculate ROC metrics
    thresholds_roc = np.arange(min(distances)-2, max(distances)+2, 0.01)
    true_positive_rate, false_positive_rate = calculate_roc_values(thresholds=thresholds_roc, distances=distances, labels=labels)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return true_positive_rate, false_positive_rate, roc_auc

def plot_roc_lfw(false_positive_rate, true_positive_rate, figure_name="roc.png"):
    """Plots the Receiver Operating Characteristic (ROC) curve.
    Args:
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
        figure_name (str): Name of the image file of the resulting ROC curve plot.
    """
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("roc_auc:",roc_auc)
    fig = plt.figure()
    plt.plot(
        false_positive_rate, true_positive_rate, color='red', lw=2, label="ROC Curve (area = {:.4f})".format(roc_auc)
    )
    data = {"fpr":false_positive_rate,"tpr":true_positive_rate,"auc":roc_auc}
    with open(os.path.splitext(figure_name)[0]+".pkl","wb") as f: pickle.dump(data,f)
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)

if __name__ == "__main__":
    distances = np.arange(0.0,1.0,0.1)
    #labels = np.where(np.random.rand((10))>0.5,True,False)
    labels = np.array([[ True,  True,  True, False, False,  True,  True, False, False,  True]])
    print(distances)
    print(labels)
    true_positive_rate, false_positive_rate, roc_auc = eval(distances=distances,labels=labels)
    # Plot ROC curve
    plot_roc_lfw(
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate
    )
