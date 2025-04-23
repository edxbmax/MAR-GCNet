import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
import pandas as pd
import os
import ast
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, f1_score


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] -= t
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(0).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(D,A), D)
    return adj

def compute_F1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_scores = []
    for i in range(len(y_pred)):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        f1 = f1_score(y_true=y_true[i], y_pred=y_pred[i])
        f1_scores.append(f1)

    return np.mean(f1_scores)

def compute_ACC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_correct, total = 0, 0
    for i in range(len(y_pred)):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i]).ravel()
        total_correct += tp + tn
        total += tn + fp + fn + tp

    return total_correct / total

def compute_TPR(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum, count = 0.0, 0
    for i, _ in enumerate(y_pred):
        y_pred[i] = np.where(y_pred[i] >= 0.5, 1, 0)
        (x, y) = confusion_matrix(y_true=y_true[i], y_pred=y_pred[i])[1]
        sum += y / (x + y)
        count += 1

    return sum / count


def compute_AUC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_auc = []
    for i in range(len(y_true[1])):
        class_auc.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    auc = roc_auc_score(y_true, y_pred)
    return auc, class_auc

# KD loss
class KdLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KdLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature

    def forward(self, outputs, labels, teacher_outputs):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / self.T, dim=1),
                                                      F.softmax(teacher_outputs / self.T, dim=1)) * (
                          self.alpha * self.T * self.T) + F.binary_cross_entropy_with_logits(outputs, labels) * (
                          1. - self.alpha)
        return kd_loss


