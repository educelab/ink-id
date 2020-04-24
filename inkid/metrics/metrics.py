from sklearn.metrics import roc_auc_score
import torch


# https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
def confusion(prediction, truth):
    _, prediction = prediction.max(1)  # Argmax
    confusion_vector = prediction.float() / truth.float()
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def recall(pred, yb):
    tp, _, _, fn = confusion(pred, yb)
    return 1 if tp + fn == 0 else tp / (tp + fn)


def precision(pred, yb):
    tp, fp, _, _ = confusion(pred, yb)
    return 1 if tp + fp == 0 else tp / (tp + fp)


def fbeta(pred, yb, beta=0.3):
    p, r = precision(pred, yb), recall(pred, yb)
    epsilon = 1e-5
    return (1 + beta**2) * (p * r) / (beta**2 * p + r + epsilon)


def accuracy(pred, yb):
    tp, fp, tn, fn = confusion(pred, yb)
    return (tp + tn) / (tp + fp + tn + fn)


def auc(pred, yb):
    _, pred = pred.max(1)  # Argmax
    return roc_auc_score(yb.cpu().numpy(), pred.cpu().numpy())
