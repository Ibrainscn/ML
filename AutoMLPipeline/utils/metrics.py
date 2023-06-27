import torch
import numpy as np
from math import sqrt
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error


class EarlyStop:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.Inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def eval_metric(params, pred, targets):
    res_metric = []
    _eval_metric = params["mlflow_runner"]["eval_metric"]
    for metric_name in _eval_metric:
        if metric_name == "mae":
            res_metric.append(mean_absolute_error(targets, pred))
        elif metric_name == "mse":
            res_metric.append(mean_squared_error(targets, pred))
        elif metric_name == "rmse":
            res_metric.append(sqrt(mean_squared_error(targets, pred)))
        elif metric_name == "pr_auc":
            res_metric.append(average_precision_score(targets, pred))
        elif metric_name == "roc_auc":
            res_metric.append(roc_auc_score(targets, pred))
        elif metric_name == "roc_auc_ovr":
            res_metric.append(roc_auc_score(targets, pred, multi_class="ovr"))
        elif metric_name == "roc_auc_ovo":
            res_metric.append(roc_auc_score(targets, pred, multi_class="ovo"))
        else:
            raise ValueError("Unknown metric")

        return res_metric
