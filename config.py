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

