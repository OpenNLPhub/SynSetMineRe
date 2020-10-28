
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





"""-------------------- Test ------------------------"""
# def test_read_embed_info():
#     NYT_embed_path =  pwd.joinpath('data','NYT','combined.embed')
#     import pdb; pdb.set_trace()
#     word2id, embed_np_matrix = read_embed_info(NYT_embed_path)


if __name__ == '__main__':
    # test_read_embed_info()
    pass
