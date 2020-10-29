'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-10-29 19:02:44
 * @desc 
'''
from pathlib import Path
from typing import Any,Dict
from dataloader import DataSet, Dataloader, DataItemSet,select_sampler
from wrapper import ModelWrapper
from model import Embedding_layer,
import config

def Test(dir_path:Path,dataconfig:Dict[Any],trainingconfig:Dict[Any]):
    dataset = DataSet(str(dir_path),dir_path.name)
    dataitems = DataItemSet(
                    dataset=dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['negative_sample_size']
                ) 
    dataloader = Dataloader(
                    dataitems=dataitems, 
                    dataset=dataset.word2id,
                    batch_size=trainingconfig['batch_size']
                )
    model = ModelWrapper()
    
