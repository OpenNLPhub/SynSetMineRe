
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-10 10:43:00
@desc [description]
'''
import numpy as np
from typing import List,Tuple
from pathlib import Path
import numpy as np
import random
import torch

pwd = Path.cwd()


def set_padding(m:List[List[int]],padding = 0)-> Tuple[np.array,np.array]:
    """Set Paddings"""
    max_len = 0
    for i in m:
        if max_len < len(i):
            max_len = len(i)
    input_ids = np.array([
        np.concatenate([ x , [padding]*(max_len - (len(x)))]) if len(x)< max_len else x for x in m
    ])
    mask = np.where(input_ids != padding,1,0)
    
    return input_ids, mask



def set_random_seed(seed = 10, deterministic = False, benchmark = False):
    """
    Args:
        seed: Random Seed
        deterministic:
                Deterministic operation may have a negative single-run performance impact, depending on the composition of your model. 
                Due to different underlying operations, which may be slower, the processing speed (e.g. the number of batches trained per second) may be lower than when the model functions nondeterministically. 
                However, even though single-run speed may be slower, depending on your application determinism may save time by facilitating experimentation, debugging, and regression testing.
        benchmark: whether cudnn to find most efficient method to process data
                If no difference of the size or dimension in the input data, Setting it true is a good way to speed up
                However, if not,  every iteration, cudnn has to find best algorithm, it cost a lot
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True

"""-------------------- Test ------------------------"""
# def test_read_embed_info():
#     NYT_embed_path =  pwd.joinpath('data','NYT','combined.embed')
#     import pdb; pdb.set_trace()
#     word2id, embed_np_matrix = read_embed_info(NYT_embed_path)


if __name__ == '__main__':
    # test_read_embed_info()
    pass
