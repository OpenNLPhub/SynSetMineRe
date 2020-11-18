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



**Distribution of Dataset**

Note:  split 1/5 train-cold.set  as dev.set

counts is the wordset size

**NYT**

| Counts    | 1                     | 2 | 3 | 4 | 5 | 6 |
| --------- | ------------------------------------- | ---- | ---- | ---- | ---- |---- |
| train_set | 18 | 1213 | 35 | 6 | 1 |0|
| test_set  | 0    | 1 | 89 | 18 | 7 |2|
| dev_set   | 0         | 227 | 20 | 6 | 1 |0|



**PubMed**


| Counts    | 1    | 2    | 3    | 4    | 5    | 6    | >6|
| --------- | ---- | ---- | ---- | ---- | ---- | ---- |  ---- |
| train_set | 0   | 19135 | 5688   | 187    | 954   | 2437   | 199    |
| test_set  | 0    | 0    | 0   | 0  | 0   | 128    | 122    |
| dev_set   | 0    | 2009  | 1556   |   1163  | 655    | 154    | 183    |



**Wiki**


| Counts    | 1    | 2    | 3    | 4    | 5    | 6    | >6|
| --------- | ---- | ---- | ---- | ---- | ---- | ---- |  ---- |
| train_set | 100   | 4243 | 12   | 6    | 1    | 0    | 0    |
| test_set  | 0    | 0    | 176   | 53   | 16    | 8    | 3    |
| dev_set   | 4    | 8555  | 12   | 6    | 1    | 0    | 0    |



## Experiment Result

（I use 20 negative sample ratio to create negative samples）

Orignal Experiment Result in Paper

|      | NYT_Test         | PubMed_Test      | Wiki_Test        |
| ---- | ---------------- | ---------------- | ---------------- |
| ARI  | 0.4491 (+0.0216) | 0.7433 (+0.0066) | 0.5643 (+0.0131) |
| NMI  | 0.9062 (+0.0153) | 0.9490 (+0.0097) | 0.9304 (+0.0023) |
| FMI  | 0.4637(+0.0192)  | 0.7445 (+0.0064) | 0.5710 (+0.0117) |



---

**split train-cold.set file into 1:4: dev_data and train_data**

Classifier Result in NYT DataSet

|                | NYT_Dev | NYT_Test | PubMed_Dev | PubMed_Test | Wiki_Dev | Wiki_Test |
| -------------- | ----------- | ------------ | ---- | ---- | ---- | ---- |
| True Positive  | 152 | 78 | 4306 | 228 |  638  |206|
| True Negative  | 2493 | 1147 | 56702 | 2481 |  8616   |2532|
| False Positive | 47 | 23 | 498 | 19 |   94  |28|
| False Negative | 102 | 39 | 1414 | 22 |   233   |50|
| Sum | 2794 | 1288 | 62920 | 2750 | 9581 |2816|



|                | NYT_Dev | NYT_Test | PubMed_Dev | PubMed_Test | Wiki_Dev |Wiki_Test|
| -------------- | ----------- | ------------ | ---- | ---- | ---- | ---- |
| Accuracy | 0.95 | 0.95 | 0.97 | 0.99 | 0.97 |0.97|
| Precision | 0.76 | 0.77 | 0.90 | 0.92 | 0.85 |0.88|
| Recall | 0.60 | 0.67 | 0.75 | 0.91 | 0.77 |0.80|
| F1-Score | 0.67 | 0.72 | 0.82 | 0.92 | 0.81 |0.84|



clustering result

|      | NYT_Train |    NYT_Test  | PubMed_Train |PubMed_Test | Wiki_Train| Wiki_Test |
| ---- | --------- | ---- | ---- | ---- | ----|-----|
| ARI | 0.16 | **0.33** | ------ | **0.69** |0.06| **0.40** |
| NMI | 0.87 | **0.85** | ------ | **0.93** |0.83| **0.88** |
| FMI | 0.23 | **0.34** | ------       | **0.70** |0.12|**0.43**|

(PubMed_Train is too big to predict, So I did not do this experiment)



---

**split train-cold.set file into 1:4: dev_data is 1/5 of train-cold.set and train_data is train-cold.set**

Classifier Result in NYT DataSet

|                | NYT_Dev | NYT_Test | PubMed_Dev | PubMed_Test | Wiki_Dev | Wiki_Test |
| -------------- | ------- | -------- | ---------- | ----------- | -------- | --------- |
| True Positive  | 1237    | 55       | 5128       | 230         | 867      | 190       |
| True Negative  | 25455   | 1154     | 56721      | 2491        | 8624     | 2541      |
| False Positive | 5       | 16       | 479        | 9           | 86       | 19        |
| False Negative | 36      | 62       | 592        | 20          | 4        | 66        |
| Sum            | 2794    | 1288     | 62920      | 2750        | 9581     | 2816      |



|           | NYT_Dev | NYT_Test | PubMed_Dev | PubMed_Test | Wiki_Dev | Wiki_Test |
| --------- | ------- | -------- | ---------- | ----------- | -------- | --------- |
| Accuracy  | 0.98    | 0.94     | 0.98       | 0.99        | 0.99     | 0.97      |
| Precision | 0.85    | 0.77     | 0.91       | 0.96        | 0.91     | 0.91      |
| Recall    | 0.92    | 0.47     | 0.90       | 0.92        | 1.00     | 0.74      |
| F1-Score  | 0.89    | 0.59     | 0.91       | 0.94        | 0.95     | 0.82      |



clustering result

|      | NYT_Train | NYT_Test | PubMed_Train | PubMed_Test | Wiki_Train | Wiki_Test |
| ---- | --------- | -------- | ------------ | ----------- | ---------- | --------- |
| ARI  | -----     | **0.42** | ------       | **0.71**    | ------     | **0.45**  |
| NMI  | -----     | **0.89** | ------       | **0.93**    | ------     | **0.90**  |
| FMI  | -----     | **0.42** | ------       | **0.71**    | ------     | **0.46**  |

(PubMed_Train is too big to predict, So I did not do this experiment)



## Experiment Detail

- I freeze the embedding layer. Only use raw word embedding to train.

- Only Use Combined.embed word vector to do experiment. Other Experiment is suspended. QAQ

  

## Result

In PubMed DataSet, I reimplement the best performance. Althought, there is a distinguish between it and result in raw paper. Compared with other method's result showed in paper, Reimplemented SysSetMine still performed better.







## Extention and Explore

I try to replace the scorer using BertModel. Omit the position_ids in bert input, using a special token to split words in one word set.



#### Embedding Problem

Many Words is [UNK] word in Bert Pretrained Model.  I try to use a classifier to map word embedding layer to bert embedding size