import torch
import math
import numpy as np


class Metrics:
    """Tracking mean metrics
    """

    def __init__(self, labels):
        """Creates an new `Metrics` instance.
        Args:
          labels: the labels for all classes.
        """

        self.labels = labels

        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.tp = 0

    def add(self, actual, predicted):
        """Adds an observation to the tracker.
        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """
        a,b=predicted.view(-1).float(),actual.view(-1).float()
        confusion = predicted.view(-1).float() / actual.view(-1).float()
        self.tn += torch.sum(torch.isnan(confusion)).item()   #fp
        self.fp += torch.sum(confusion == float("inf")).item()  #tn
        self.fn += torch.sum(confusion == 0).item() #fn
        self.tp += torch.sum(confusion == 1).item() #tp   fn  fp

    def get_precision(self):

        return self.tp / (self.tp + self.fp)

    def get_recall(self):

        return self.tp / (self.tp + self.fn)

    def get_f_score(self):

        pr = 2 *(self.tp / (self.tp + self.fp)) * (self.tp / (self.tp + self.fn))
        p_r = (self.tp / (self.tp + self.fp)) + (self.tp / (self.tp + self.fn))
        return pr / p_r

    def get_oa(self):
        
        t_pn = self.tp + self.tn
        t_tpn = self.tp + self.tn + self.fp + self.fn
        return t_pn / t_tpn

    def get_miou(self):
        """Retrieves the mean Intersection over Union score.
        Returns:
          The mean Intersection over Union score for all observations seen so far.
        """
        return np.nanmean([self.tn / (self.tn + self.fn + self.fp), self.tp / (self.tp + self.fn + self.fp)])

    def get_fg_iou(self):
        """Retrieves the foreground Intersection over Union score.
        Returns:
          The foreground Intersection over Union score for all observations seen so far.
        """

        try:
            iou = self.tp / (self.tp + self.fn + self.fp)
        except ZeroDivisionError:
            iou = float("Inf")

        return iou

    def get_kappa(self):
        """=(OA-P)/(1-P)   P=[(TP+FP)(TP+FN)+(FN+TN)(TP+TN)]/(TP+FP+TN+FN)的平方
        """
        try:
            oa=self.get_oa()
            p=((self.tp+self.fp)*(self.tp+self.fn)+(self.fn+self.tn)*(self.tp+self.tn))/math.pow((self.tp+self.fp+self.tn+self.fn),2)
            kappa=(oa-p)/(1-p)
        except ZeroDivisionError:
            kappa = float("Inf")
        return kappa


    def get_mcc(self):
        """Retrieves the Matthew's Coefficient Correlation score.
        Returns:
          The Matthew's Coefficient Correlation score for all observations seen so far.
        """

        try:
            mcc = (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
                (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
            )
        except ZeroDivisionError:
            mcc = float("Inf")

        return mcc

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count