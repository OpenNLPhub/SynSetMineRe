
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-10 10:41:30
@desc 
    : Generate DataSet and DataLoader for Classifier
'''

from pathlib import Path
import random
import re
from re import sub
from typing import Dict, List, Optional, Set, Tuple, Any
from copy import copy
import numpy as np
from numpy.core.defchararray import index
from torch.utils import data
from utils import set_padding
import pickle
pattern = "(?<=\')[^\|\']*\|\|[^\|\']*?(?=\')"




class DataSet(object):
    """class: description of Raw Data"""
    def __init__(self,file_path:Path,name:str = "DataSet",word_emb_select:Optional[str] = None):
        self.name = name
        self.vocab,self.raw_sets,self.max_set_size,self.min_set_size,self.average_set_size = self._initilize(file_path=str(file_path))
        word_emb_file = None
        if word_emb_select == None:
            word_emb_file = file_path.joinpath('combined.embed')
        elif word_emb_select == 'fastText-subword':
            word_emb_file = file_path.joinpath('combined.fastText-with-subword.embed')
        elif word_emb_select == 'fastText-no-subword':
            word_emb_file = file_path.joinpath('combined.fastText-no-subword.embed')
        
        self.word2id, self.embedding_vec = self.read_embed_info(str(word_emb_file))
      
    def __iter__(self):
        for word_set in self.raw_sets:
            yield word_set
    
    def __repr__(self):
        s="-------------Description of Dataset-------------\n"
        return s+'Raw DataSet Name {} \n vocab size {} \n num of sets {} \n the size of biggest set {} \n the size of smallest set {} \n the average size of all sets {}'.format(\
                self.name,len(self.vocab),len(self.raw_sets),self.max_set_size,self.min_set_size,self.average_set_size)

        
    def _initilize(self,file_path:str)->None:
        '''Initialize Dataset from raw string
            @Param file_path:Raw Data file path
            @Return
                vocab: vocabulary for all word in dataset 
                type: dict {'word':'cluster_id'}

                allsets: Graund Trueth clusters
                type: list [ ["U.S.A","U.S"] ,[]]

                max_set_size: maximun of set size in allsets
                type: int

                min_set_size: minimun of set size in allsets
                type: int

                average_set_size: average of set size in allsets
                type: float
        '''
        vocab = {}
        allsets = []
        max_set_size = -1
        min_set_size = 1e10
        sum_set_size = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            it = re.finditer(pattern, line)
            item = []
            size = 0.
            for i in it:
                size += 1
                word, cluster = line[i.start():i.end()].split("||")
                vocab[word] = cluster
                item.append(word)
            max_set_size = max_set_size if max_set_size>size else size
            min_set_size = min_set_size if min_set_size<size else size
            sum_set_size += size
            allsets.append(item)

        return vocab,allsets,max_set_size,min_set_size,sum_set_size/len(allsets)
    
    def read_embed_info(self,filepath:str):
        """Read word embeding file"""
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
    



class DataItemSampler(object):
    """Interface of various sampler"""
    def sample(self,wordpool:Dict, wordset:List[str], negative_sample_size:int)->Tuple[List[Tuple],int]:
        raise NotImplementedError
    
    def _negative_sample(self,wordpool: Dict, wordset: List[str], negative_sample_size:int)-> List[str]:
        pos_cluster = wordpool[wordset[0]]
        sample_word_pool = [ word for word in wordpool.keys() if wordpool[word] != pos_cluster]
        #sample size bigger than sample pool
        if negative_sample_size > len(sample_word_pool):
            return sample_word_pool
        else:
            return random.sample(sample_word_pool,negative_sample_size)


class Sample_size_repeat_size(DataItemSampler):
    """A sample method to sample dataitem : For one original word set, randomly get subset size, and repeat this size. 
        This is the strategy to original AAAI submission."""
    def __init__(self) -> None:
        super(Sample_size_repeat_size,self).__init__()

    def sample(self,wordpool:Dict, wordset:List[str], negtive_sample_size:int)->Tuple[List[Tuple],int]:
        ans = []
        setsize = len(wordset)

        if setsize == 1:
            # for only one word set, we generate one positive item and one negative item
            pos_word = wordset[0]
            neg_word = random.choice(list(wordpool.keys()))
            while neg_word == pos_word:
                neg_word = random.choice(wordpool.keys())
            ans = [
                (pos_word,pos_word,1),
                (pos_word,neg_word,0)
            ]
            return ans,1,1,1

        # random choice subsize
        new_set =  wordset.copy()
        subset_size = random.choice(range(1,setsize))
        
        pos_word_set = new_set[:subset_size]
        pos_word = new_set[subset_size]

        ans = [(pos_word_set, pos_word, 1)]

        neg_word_list = self._negative_sample(wordpool,wordset,negtive_sample_size)
        for neg_word in neg_word_list:
            ans.append((pos_word_set,neg_word,0))
        pos_item_nums = 1
        neg_item_nums = pos_item_nums * negtive_sample_size
        return ans,subset_size, pos_item_nums, neg_item_nums

class Sample_vary_size_enumerate(DataItemSampler):
    """A sample method to sample dataitem : For one original word set, enunmerate all subset size and get item
    """
    def __init__(self) -> None:
        super(Sample_vary_size_enumerate,self).__init__()

    def sample(self, wordpool: Dict, wordset: List[str], negative_sample_size:int) -> Tuple[List[Tuple],int]:
        """c"""
        pass


def select_sampler(name:str)-> DataItemSampler:
    if name == "sample_vary_size_enumerate":
        return Sample_vary_size_enumerate()
    else:
        return Sample_size_repeat_size()

class DataItemSet(object):
    """DataItemSet  Generate Training and Testing data item"""
    def __init__(self, dataset:DataSet, sampler:DataItemSampler, negative_sample_size:int = 10)->None:
        self.negative_sample_size = negative_sample_size
        self.vocab = dataset.vocab
        self.sampler = sampler
        self.average_set_size = -1
        self.max_set_size = -1
        self.min_set_size = -1
        self.dataitems = self._initialize(dataset)

        '''
        item in dataitems is a tuple
        (
            'Set': [ 'word1' , 'word3' ,..., 'wordn']
            'word' : word (waiting to classified)
            'label' : 1 or 0 (positive item or negative item)
        )
        '''

    def _initialize(self,dataset:DataSet)->None:
        dataitem = []
        subset_size_list = []
        neg_item_num = 0
        pos_item_num = 0
        for wordset in dataset:
            subitems, subset_size, pos_item_size, neg_item_size = self.sampler.sample(self.vocab,wordset,self.negative_sample_size)

            neg_item_num += neg_item_size
            pos_item_num += pos_item_size

            dataitem.extend(subitems)
            subset_size_list.append(subset_size)
        
        self.pos_item_num = pos_item_num
        self.neg_item_num = neg_item_num
        self.average_set_size = 1.0 * sum(subset_size_list)/len(subset_size_list)
        self.max_set_size = max(subset_size_list)
        self.min_set_size = min(subset_size_list)
        return dataitem

    def __repr__(self)->None:
        s="-----------Description of DataItemSet--------------\n"
        return s+'nums of dataitem {} \n the size of biggest set item {} \n the size of smallest set item {} \n the average size of all sets item {:.2f} \n negtive items in dataset:{} \n positive items in dataset:{}'.format(\
                len(self.dataitems),self.max_set_size,self.min_set_size,self.average_set_size,self.neg_item_num, self.pos_item_num)

    def __len__(self) -> int:
        return len(self.dataitems)
    
    def __iter__(self)->None:
        for i in self.dataitems:
            yield i
        
    def __getitem__(self,index):
        return self.dataitems[index]

    @classmethod
    def save(cls,path:Path):
        """save dataitem"""
        with open(path, 'wb', encoding='utf-8') as f:
            pickle.dump(cls,f)
        # pass

    @classmethod
    def load(cls,path:Path):
        """load dataitem"""
        with open(path, 'rb', encoding='utf-8') as f:
            ans = pickle.load(f)
        return ans


class Dataloader(object):
    """Dataloader to get batch size dataitem"""
    def __init__(self, dataitems:DataItemSet, word2id, batch_size:int) -> None:
        self.data = dataitems
        self.l = len(self.data)/batch_size if len(self.data)%batch_size == 0 else len(self.data)/batch_size + 1
        self.batch_size = batch_size
        self.word2id = word2id
    def __len__(self):
        return self.l

    def __iter__(self):
        ixs = list(range(0,len(self.data)))
        random.shuffle(ixs)
        batch_old_word_set, batch_new_word_set, batch_label = [], [], []
        for index,ix in enumerate(ixs):
            word_set, waiting_word, label = self.data[ix]
            word_id_set = [ self.word2id[word] for word in word_set]
            waiting_word_id = self.word2id[waiting_word]
            new_word_id_set = word_id_set.copy()
            new_word_id_set.append(waiting_word_id)
            
            batch_old_word_set.append(word_id_set)
            batch_new_word_set.append(new_word_id_set)
            batch_label.append(label)

            if (index+1) % self.batch_size == 0 or index == len(self.data) - 1:
                (batch_old_word_set_, old_mask), (batch_new_word_set_,new_mask) = set_padding(batch_old_word_set), set_padding(batch_new_word_set)
                # batch_label = self._trans_label(batch_label)
                yield batch_old_word_set_, old_mask, batch_new_word_set_, new_mask, batch_label
                batch_old_word_set, batch_new_word_set, batch_label = [],[],[]
    
    
    def split(self, q:float=0.2) -> Tuple[Any]:
        dataitem = copy(self.data)
        random.shuffle(dataitem)
        sub_l = int(self.l * q)
        return [
                Dataloader(dataitem[:sub_l], self.word2id, self.batch_size),
                Dataloader(dataitem[sub_l:], self.word2id, self.batch_size)
            ]
        








'''------------------------------ Test ---------------------------------- '''

def test_dataset():
    from pathlib import Path
    pwd = Path.cwd()
    NYT_dir_path = Path.joinpath(pwd,'data','NYT')
    print(NYT_dir_path)
    train_data_NYT = DataSet(Path.joinpath(NYT_dir_path,'train-cold.set'),'NYT-Train')
    print(train_data_NYT)
    Med_dir_path = Path.joinpath(pwd,'data','PubMed')
    train_data_MedPub = DataSet(Path.joinpath(Med_dir_path,'train-cold.set'),'PubMed-Train')
    print(train_data_MedPub)

    Wiki_dir_path = Path.joinpath(pwd,'data','Wiki')
    train_data_Wiki = DataSet(Path.joinpath(Wiki_dir_path,'train-cold.set'),'Wiki-Train')
    print(train_data_Wiki)

def test_dataitemset():
    from pathlib import Path
    pwd = Path.cwd()
    NYT_dir_path = Path.joinpath(pwd,'data','NYT')
    train_data_NYT = DataSet(Path.joinpath(NYT_dir_path,'train-cold.set'),'NYT-Train')
    sampler = Sample_size_repeat_size()
    negative_sample_size = 10
    dataitem = DataItemSet(train_data_NYT, sampler, negative_sample_size)
    print(dataitem)
    # for i in dataitem:
    #     pass

def test_dataloader():
    from pathlib import Path
    pwd = Path.cwd()
    NYT_dir_path = Path.joinpath(pwd,'data','NYT')
    train_data_NYT = DataSet(Path.joinpath(NYT_dir_path,'train-cold.set'),'NYT-Train')
    sampler = Sample_size_repeat_size()
    negative_sample_size = 10
    dataitem = DataItemSet(train_data_NYT, sampler, negative_sample_size)
    batch_size = 32

    word2id = train_data_NYT.word2id
    dataloader = Dataloader(dataitems= dataitem, word2id= word2id, batch_size= batch_size)
    
    for item in dataloader:
        batch_old_word_set, old_mask, batch_new_word_set, new_mask, batch_label = item
        import pdb;pdb.set_trace()
    

if __name__ == '__main__':
    test_dataset()
    test_dataitemset()
    test_dataloader()

