'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-10-26 16:36:19
 * @desc 
'''
import itertools
from typing import Any, Dict, Optional, Sequence, Tuple, overload
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import recall_score
from torch.backends.mkldnn import set_flags
import numpy as np
from scipy.special import comb

class EvalUnit(object):
    """Smalled Evaluating metrics unit
    Attribute:
        tp: True Positive item
        fp: False Positive item
        fn: False Negative item
        tn: True Negative item
        name: A label to describe specific instance
    """

    def __init__(self, tn:int=0, fp:int=0, fn:int=0, tp:int=0, name:Optional[int]=None) -> None:
        super(EvalUnit,self).__init__()
        self.name = name if name is not None else "None"
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn

    @overload
    def __repr__(self):
        desc = '--------- Desc EvalUnit {}---------'.format(self.name)
        desc += 'True Positive:{} \nTrue Negative:{} \nFalse Positive:{} \nFalse Negative:{} \n'.format(
                self.tp, self.tn, self.fp, self.fn
                )
        desc += 'Accuracy:{:.2f} \nPrecision:{:.2f} \nRecall:{:.2f} \nF1-Score:{:.2f}'.format(
                self.accuracy(), self.precision(), self.recall(), self.f1_score()
                )
        return desc

    @overload
    def __add__(self,other) -> "EvalUnit":
        return EvalUnit(
            self.tn + other.tn,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tp + other.tp,
            )
        
    @overload
    def __iadd__(self,other) -> "EvalUnit":
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        self.tp += other.tp
        return self

    def accuracy(self) -> float:
        return float(self.tn + self. tp) / (self.fp + self.fn + self.tn + self.tp)

    def f1_score(self) -> float:
        r = self.recall()
        p = self.precesion
        return 2 * r * p / (p + r) if p + r != 0  else 0.

    def precision(self) -> float:
        return float(self.tp) / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0.
    
    def recall(self) -> float:
        return float(self.tp) / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0.

    def metrics(self) -> Tuple[float]:
        return (self.accuracy, self.precision, self.recall, self.f1_score)


def binary_confusion_matrix_evaluate(y_true:Sequence[Any], y_pred:Sequence[Any]) -> EvalUnit:
    # import pdb; pdb.set_trace()
    tn ,fp, fn, tp =  confusion_matrix(y_true,y_pred).ravel()
    return EvalUnit(tn,fp,fn,tp)


""" ---------------- cluster Metrics ---------------- """
def rand_index_plus(pred_cluster:Dict[Any], target_cluster:Dict[Any]) -> EvalUnit:
    """Calcultae the rand index and Get detailed result
    RI = Accuracy = (TP+TN)/(TP,TN,FP,FN)
    Args:
        pred_cluster: Dict element:cluster_id （cluster_id from 0 to max_size）| predicted clusters 
        target_cluster: Dict element:clutser_id （cluster_id from 0 to max_size) | target clusters  
    Returns:
        In order to return detailed data, It will return a EvalUnit, 
        call accuracy to get RI.
    """
    pred_elements = list(pred_cluster.keys())
    target_elements = list(target_cluster.keys())
    it = itertools.product(pred_elements,target_elements)
    tp,fp,tn,fn = 0,0,0,0
    for x,y in it:
        if x != y:#other word
            x_cluster = pred_cluster[x]
            x_cluster_ = target_cluster[x]
            y_cluster = pred_cluster[y]
            y_cluster_ = target_cluster[y]

            if x_cluster == y_cluster and x_cluster_ == y_cluster_:
                tp += 1
            elif x_cluster != y_cluster and x_cluster_ != y_cluster_:
                tn += 1
            elif x_cluster == y_cluster and x_cluster_ != y_cluster_:
                fp += 1
            else:
                fn +=1
    return EvalUnit(tp,tn,fp,fn,'rand_index')


def rand_index(pred_cluster: Dict[Any], target_cluster: Dict[Any]) -> float:
    """Use contingency_table to get RI directly
    Args:
        pred_cluster: Dict cluster_id: List[element] （cluster_id from 0 to max_size）| predicted clusters 
        target_cluster: Dict cluster_id: List[element] （cluster_id from 0 to max_size) | target clusters  
    Return:
        RI (float)
    """
    pred_cluster_size = len(pred_cluster)
    target_cluster_size = len(target_cluster)
    contingency_table = np.zeros((pred_cluster_size,target_cluster_size))
    
    for i, p_cluster in enumerate(pred_cluster):
        for j, t_cluster in enumerate(target_cluster):
            #find common element
            l = [*p_cluster,*t_cluster]
            contingency_table[i][j] = len(l) - len(set(l))
    s = comb(np.sum(contingency_table), 2)
    a = 0
    for i in np.nditer(contingency_table):
        a += comb(i,2)
    return a/s

def adjusted_rand_index(pred_cluster:Dict[Any], target_cluster:Dict[Any]):
    """Docstring
    Using Contingency Matrix to calculate ARI
    Continggency Matrix
    --------------------------------
    XY  | Y_1  Y_2  ...  Y_s  | sums
    --------------------------------
    X_1 | n_11 n_12 ...  n_1s | a_1
    X_2 | n_21 n_22 ...  n_2s | a_2
    ... | ...  ...  ...  ...  | ...
    X_r | n_r1 n_r2 ...  n_rs | a_r
    sum | b_1  b_2  ...  b_s  |
    --------------------------------
    f(x) = comb(x,2)
    ARI = [ sum f(n_ij) - sum f(a_ij) * sum f(b_ij) / f(n) ] /
            [0.5 * [ sum f(a_ij) + sum f(b_ij)] - sum f(a_ij) * sum f(b_ij) / f(n)]
    """
    pred_cluster_size = len(set(pred_cluster.values()))
    target_cluster_size = len(set(target_cluster.values()))
    contingency_table = np.zeros((pred_cluster_size, target_cluster_size))
    s = comb(np.sum(contingency_table), 2)
    ij = 0
    for i in np.npiter(contingency_table):
        ij += comb(i,2)
    pred_sum = np.sum(contingency_table, axis=1)
    target_sum = np.sum(contingency_table, aixs=0)
    
    pred_comb_sum = 0
    for i in np.npiter(pred_sum):
        pred_comb_sum += comb(i,2)
    target_comb_sum = 0
    for i in np.npiter(target_sum):
        target_comb_sum += comb(i,2)
    tmp = pred_comb_sum * target_comb_sum / s
    ARI = (ij - tmp) / (0.5*(pred_comb_sum+target_comb_sum) - tmp)
    
    return ARI



if __name__ == '__main__':
    print(1)