

from typing import List
from collections import defaultdict
from sklearn import metrics as sk_metrics
import numpy as np
from matplotlib import pyplot as plt

class Evaluation():
    def __init__(self, y, out : List[float] ):
        self.y = y
        self.out = out
        self.threshold = 0.5
        self.outlabel = np.array(out > 0.5).astype(int)
        self.cfm = lambda y,y_hat : sk_metrics.confusion_matrix(y,y_hat).flatten()
    @property
    def confusion_matrix(self):
        return sk_metrics.confusion_matrix(self.y,self.outlabel)

    @property
    def rate(self): return self._rate
    
    @property
    def rate_curve(self): return self._rate(None)
    
    def _rate(self, threshold : float = 0.5):
        rate = defaultdict(list)
        if threshold is None:

            for i in np.arange(0,1,0.05):
                q = np.array(self.out > i).astype(int)
                tn,fp,fn,tp = self.cfm(self.y,q)
                rate['tpr'] += [tp/ (tp+fn)]
                rate['fpr'] += [fp/ (tn+fp)]
                rate['fnr'] += [fn/ (tp+fn)]
                rate['tnr'] += [tn/ (tn+fp)]
                rate['acc'] += [(tn+tp)/(fp+fn+tp+tn)]
                rate['precision'] += [tp/(tp+fp)]
                rate['f1'] += [sk_metrics.f1_score(self.y,q)] #harmonic mean between precision and recall
                rate['cohenkappa'] += [sk_metrics.cohen_kappa_score(self.y,q)]

            
            return rate
        else:
            tn,fp,fn,tp = self.confusion_matrix.flatten()
            rate['tpr'] = tp/ (tp+fn)
            rate['fpr'] = fp/ (tn+fp)
            rate['fnr'] = fn/ (tp+fn)
            rate['tnr'] = tn/ (tn+fp)
            rate['acc'] = (tn+tp)/(fp+fn+tp+tn)
            rate['precision'] = tp/(tp+fp)
            rate['AP'] = sk_metrics.average_precision_score(self.y , self.out)
            rate['f1'] = sk_metrics.f1_score(self.y,self.outlabel)
            rate['cohenkappa'] = sk_metrics.cohen_kappa_score(self.y,self.outlabel)
            rate['auroc'] = self.roc
            return rate
    
    
    @property
    def roc(self):
        
        #prob that model ranks positive example more highly than  a random negative, - An introduction to ROC analysis, Tom Fawcett 2005
        ## classes is heavily imbalanced shouldnt use this, false positive rate is pulled down by number of negatives
        
        x,y,threshold = sk_metrics.roc_curve(self.y, self.out, pos_label=1)
        auc = sk_metrics.roc_auc_score(self.y, self.outlabel) #should be proba, but for paper this is still not changed
        plt.plot(np.arange(0,1,0.02),np.arange(0,1,0.02), linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        plt.plot(x,y)
        plt.legend(['Diagonal',f'AUC : {auc}']) 
        return auc
    
    @property
    def prec_recall(self):
        return sk_metrics.precision_recall_curve(self.y, self.out)
    

    def plot_rate(self):
        rate = self.rate_curve
        _, ax = plt.subplots()

## The Precision-Recall Plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets
#, Takaya Satio, Marc Rehmsmeier 2015 , PR ROC comparison also found in Introduction ROC to analysis

# 1. ROC is same for both balanced imbalanced, usually more negative class lead to lower FPR, but that doesnt mean model better
# in efficient roc algorithm, (FP/N , TP/P) is used , TP = TP + 1 if positive, TP/P can be very high in early retrieval in  imbalanced dataset
# 2.PR curve focus on positive class, care less about frequent negative
# Question :: if model is bad, isnt both ROC and PR will be bad? TPR is bad, FPR high, precision bad
#(yes) but in an imbalanced dataset, your TN high doesnt mean model draw good line in the space, FN is not so common as positive example is lesser
# using PR curve, we focus on how precise is the model, fp is also included but tn is not, if fp is high Precision will not be good,
# but FPR might be ok since TN is high