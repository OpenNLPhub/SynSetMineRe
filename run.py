'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-10-29 19:02:44
 * @desc 
'''
from typing import Any,Dict
from dataloader import DataSetDir, DataSet, Dataloader, DataItemSet,select_sampler
from wrapper import ModelWrapper
from model import Embedding_layer, Scorer, SetinstanceClassifier
from evaluate import select_evaluate_func
import config 
from config import TrainingConfig,OperateConfig,DataConfig,ModelConfig
from log import logger

def test_clustertask(operateconfig:Dict,dataconfig:Dict, trainingconfig:Dict, modelconfig:Dict):
    
    dir_path =  dataconfig['data_dir_path']

    if not dir_path:
        raise KeyError

    datasetdir = DataSetDir(dir_path)
    train_datasetitem = DataItemSet(
                    dataset=datasetdir.train_dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['negative_sample_size']
                ) 
    test_datasetitem = DataItemSet(
                    dataset=datasetdir.test_dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['test_negative_sample_size']
                )
    dev_datasetitem = DataItemSet(
                    dataset=datasetdir.dev_dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['test_negative_sample_size']
                )

    train_dataloader = Dataloader(
                    dataitems=train_datasetitem, 
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
    test_dataloader = Dataloader(
                    dataitems=test_datasetitem,
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
    dev_dataloader = Dataloader(
                    dataitems=dev_datasetitem,
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )

    # combine model
    embedding_layer = Embedding_layer.from_pretrained(datasetdir.embedding_vec)
    embedding_layer.freeze_parameters()
    scorer = Scorer(
                embedding_layer, 
                modelconfig['embed_trans_hidden_size'],
                modelconfig['post_trans_hidden_size'],
                modelconfig['dropout']
            )
    model = SetinstanceClassifier(
                scorer=scorer,
                name=modelconfig['name'],
                version=modelconfig['version']
            )
    
    wrapper = ModelWrapper(model,trainingconfig)
    
    if operateconfig['resume']:
        wrapper.load_check_point()
        # continue to trainning

    if operateconfig['train']:
        wrapper.train(train_dataloader=train_dataloader,dev_dataloader=dev_dataloader)

    if operateconfig['test']:
        wrapper.test(test_dataloader=test_dataloader)

    if operateconfig['predict']:
        # func_list = select_evaluate_func(operateconfig['eval_function'])

        pred_word_set = wrapper.cluster_predict(
                    dataset=datasetdir.test_dataset,
                    word2id=datasetdir.word2id,
                    outputfile=trainingconfig['result_out_dir'].joinpath(datasetdir.name+'_result.txt')
                )
        ans = wrapper.evaluate(datasetdir.train_dataset, pred_word_set,function_list=func_list)
        logger.info("{} DataSet Cluster Prediction".format(datasetdir.train_dataset.name))
        for name,f in ans:
            logger.info("{} : {:.2f}".format(name,f))


def NYT():
    DataConfig['data_dir_path'] = config.NYT_DIR_PATH
    test_clustertask(OperateConfig,DataConfig,TrainingConfig,ModelConfig)

if __name__ == '__main__':
    NYT()
    
    
    
