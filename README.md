# SynSetMineRe
Reimplemtation of SynSetMine proposed in Mining Entity Synonyms with Efficient Neural Set Generation


## DataSet Description

**Training Data Raw Word Set**

|                  | NYT  | Wiki | PubMed |
| ---------------- | ---- | ---- | ------ |
| Vocab Size       | 2573 | 8613 | 72623  |
| Sets Nums        | 1273 | 4359 | 28600  |
| Max Set sSize    | 5.0  | 3.0  | 19.0   |
| Min Set Size     | 1.0  | 1.0  | 1.0    |
| Average Set Size | 2.03 | 2.00 | 2.53   |



## Experiment Result

（I use 20 negative sample ratio to create negative samples）

Classifier Result in NYT DataSet

|                | NYT_Dev | NYT_Test | PubMed_Dev | PubMed_Test | Wiki_Dev | Wiki_Test |
| -------------- | ----------- | ------------ | ---- | ---- | ---- | ---- |
| True Positive  | 152 | 78 |      |      |      ||
| True Negative  | 2493 | 1147 |      |      |      ||
| False Positive | 47 | 23 |      |      |      ||
| False Negative | 102 | 39 |      |      |      ||
| Sum | 2794 | 1288 |      |      |      ||

|                | NYT_Dev | NYT_Test | PubMed_Dev | PubMed_Test | Wiki_Dev |
| -------------- | ----------- | ------------ | ---- | ---- | ---- |
| Accuracy | 0.95 | 0.95 |      |      |      |
| Precision | 0.76 | 0.77 |      |      |      |
| Recall | 0.60 | 0.67 |      |      |      |
| F1-Score | 0.67 | 0.72 |      |      |      |

