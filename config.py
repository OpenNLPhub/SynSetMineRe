'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-10-28 16:39:46
 * @desc 
'''
import os
from pathlib import Path
cwd = Path.cwd()

""" ------------- Path Config ------------- """
NYT_DIR_PATH = Path.joinpath(cwd,'data','NYT')
PubMed_DIR_PATH = Path.joinpath(cwd, 'data', 'PubMed')
Wiki_DIR_PATH = Path.joinpath(cwd, 'data', 'Wiki')

""" ---------------- Own Config ----------- """
#default training Config
TrainingConfig ={
    'loss_fn' : 'crossEntropy',
    'threshold' :  0.5,
    'epoch' : 500,
    'checkpoint_epoch' : 5,
    'print_step' : 15,
    'lr' : 1e-4,
    'checkpoint_dir' : cwd.joinpath('checkpoint'),
    'batch_size' : 32,
    'result_out_dir' : cwd.joinpath('result')
}

#default Operate Config
OperateConfig = {
    'resume': False,
    'train' : True,
    'test' : True,
    'predict' : True,
    'eval_function':['ARI','NMI','FMI']
}

#default dataconfig
DataConfig = {
    'data_dir_path' : None,
    'sample_strategy' : 'sample_size_repeat_size',
    'negative_sample_size' : 10,
    'test_negative_sample_size' : 2
}

#default modelconfig
ModelConfig = {
    'name' : 'SynSetMine',
    'version' : 'v1.0.0',
    'embed_trans_hidden_size' : 250,
    'post_trans_hidden_size' : 500,
    'dropout' : 0.3
}

