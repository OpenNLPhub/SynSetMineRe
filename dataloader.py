
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
from typing import Dict, List, Set, Tuple
import copy
from utils import set_padding, read_embed_info
import pickle
pattern = "(?<=\')[^\|\']*\|\|[^\|\']*?(?=\')"




class DataSet(object):
    """class: description of Raw Data"""
    def __init__(self,file_path:str,name:str):
        self.name = name

        self.vocab,self.raw_sets,self.max_set_size,self.min_set_size,self.average_set_size = self._initilize(file_path=file_path)
        
      
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
            return ans,1

        # random choice subsize
        new_set =  wordset.copy()
        subset_size = random.choice(range(1,setsize))
        
        pos_word_set = new_set[:subset_size]
        pos_word = new_set[subset_size]

        ans = [(pos_word_set, pos_word, 1)]

        neg_word_list = self._negative_sample(wordpool,wordset,negtive_sample_size)
        for neg_word in neg_word_list:
            ans.append((pos_word_set,neg_word,0))
        
        return ans,subset_size

class Sample_vary_size_enumerate(DataItemSampler):
    """A sample method to sample dataitem : For one original word set, enunmerate all subset size and get item
    """
    def __init__(self) -> None:
        super(Sample_vary_size_enumerate,self).__init__()

    def sample(self, wordpool: Dict, wordset: List[str], negative_sample_size:int) -> Tuple[List[Tuple],int]:
        """c"""
        pass






class DataItemSet(object):
    """DataItemSet  Generate Training and Testing data item"""
    def __init__(self, dataset:DataSet, sampler:DataItemSampler, negative_sample_size:int)->None:
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
        for wordset in dataset:
            subitems,subset_size = self.sampler.sample(self.vocab,wordset,self.negative_sample_size)

            dataitem.extend(subitems)
            subset_size_list.append(subset_size)
        self.average_set_size = 1.0 * sum(subset_size_list)/len(subset_size_list)
        self.max_set_size = max(subset_size_list)
        self.min_set_size = min(subset_size_list)
        return dataitem

    def __repr__(self)->None:
        s="-----------Description of DataItemSet--------------\n"
        return s+'nums of dataitem {} \n the size of biggest set item {} \n the size of smallest set item {} \n the average size of all sets item {}'.format(\
                len(self.dataitems),self.max_set_size,self.min_set_size,self.average_set_size)
    
    def __len__(self) -> int:
        return len(self.dataitems)
    
    def __iter__(self)->None:
        for i in self.dataitems:
            yield i

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
        ixs = range(0,len(self.data))
        ixs = random.shuffle(ixs)
        batch_old_word_set, batch_new_word_set, batch_label = [], [], []
        for index,ix in enumerate(ixs):
            word_set, waiting_word, label = self.data[ix]
            word_id_set = [ self.word2id(word) for word in word_set]
            waiting_word_id = self.word2id(waiting_word)
            new_word_id_set = copy(word_id_set)
            new_word_id_set.append(waiting_word_id)
            
            batch_old_word_set.append(word_id_set)
            batch_new_word_set.append(new_word_id_set)
            batch_label.append(label)

            if (index+1) % self.batch_size == 0 or index == len(self.data-1):
                batch_old_word_set_, batch_new_word_set_ = set_padding(batch_old_word_set), set_padding(batch_new_word_set)
                yield batch_old_word_set, batch_new_word_set, batch_label
                batch_old_word_set, batch_new_word_set, batch_label = [],[],[]
        






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
    #     import pdb;pdb.set_trace()

def test_dataloader():
    from pathlib import Path
    pwd = Path.cwd()
    NYT_dir_path = Path.joinpath(pwd,'data','NYT')
    train_data_NYT = DataSet(Path.joinpath(NYT_dir_path,'train-cold.set'),'NYT-Train')
    sampler = Sample_size_repeat_size()
    negative_sample_size = 10
    dataitem = DataItemSet(train_data_NYT, sampler, negative_sample_size)
    batch_size = 32

    _, word2id = read_embed_info(Path.joinpath(NYT_dir_path,'combined.embed'))
    dataloader = Dataloader(dataitems= dataitem, word2id= word2id, batch_size= batch_size)
    
    for item in dataloader:
        import pdb;pdb.set_trace()
        
if __name__ == '__main__':
    # test_dataset()
    test_dataitemset()
