'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-10-26 16:36:19
 * @desc 
'''
from os import name
from typing import Any, Optional, Sequence, Tuple, overload
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import recall_score
from torch.backends.mkldnn import set_flags


class EvalUnit(object):
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
    

if __name__ == '__main__':
    print(1)