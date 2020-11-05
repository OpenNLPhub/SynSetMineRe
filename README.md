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
| True Positive  | 152 | 78 |      |      |  638  |206|
| True Negative  | 2493 | 1147 |      |      |  8616   |2532|
| False Positive | 47 | 23 |      |      |   94  |28|
| False Negative | 102 | 39 |      |      |   233   |50|
| Sum | 2794 | 1288 |      |      | 9581 |2816|



|                | NYT_Dev | NYT_Test | PubMed_Dev | PubMed_Test | Wiki_Dev |Wiki_Test|
| -------------- | ----------- | ------------ | ---- | ---- | ---- | ---- |
| Accuracy | 0.95 | 0.95 |      |      | 0.97 |0.97|
| Precision | 0.76 | 0.77 |      |      | 0.87 |0.88|
| Recall | 0.60 | 0.67 |      |      | 0.73 |0.80|
| F1-Score | 0.67 | 0.72 |      |      | 0.80 |0.84|



clustering result

|      | NYT_Train |    NYT_Test  | PubMed_Train |PubMed_Test | Wiki_Train| Wiki_Test |
| ---- | --------- | ---- | ---- | ---- | ----|-----|
| ARI | 0.16 |      | | |0.06| |
| NMI | 0.87 |      | | |0.83| |
| FMI | 0.23 |      | | |0.12||

