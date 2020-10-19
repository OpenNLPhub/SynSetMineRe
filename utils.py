
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-10 10:43:00
@desc [description]
'''
import numpy as np
from typing import List
from pathlib import Path
import numpy as np

pwd = Path.cwd()

def read_embed_info(filepath:Path):
    """ """
    with open(filepath, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    line_zero = lines[0]
    vocab_size, dim_size = [ int(i) for i in line_zero.strip().split(' ')]
    word2id = {}
    word2id['PAD'] = 0
    word2id['UNK'] = 1
    embed_matrix = [[0]*dim_size,[0]*dim_size]
    for line in lines[1:]:
        t = line.strip().split(' ')
        word, _ = t[0].split('||')
        word2id[word] = len(word2id)
        nums = [ eval(i) for i in t[1:]]
        embed_matrix.append(nums)
    
    embed_np_matrix = np.array(embed_matrix)
    return word2id, embed_np_matrix

def set_padding(m:List[List[int]],padding = 0)-> np.array:
    max_len = 0
    for i in m:
        if max_len < len(i):
            max_len = m
    input_ids = np.array([
        np.concatenate([ x , [padding]*(max_len - (len(x)))]) if len(x)< max_len else x for x in m
    ])
    mask = np.where(input_ids != padding,1,0)
    
    return input_ids, mask

def test_read_embed_info():
    NYT_embed_path =  pwd.joinpath('data','NYT','combined.embed')
    import pdb; pdb.set_trace()
    word2id, embed_np_matrix = read_embed_info(NYT_embed_path)


if __name__ == '__main__':
    test_read_embed_info()
