
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-20 15:58:45
@desc [description]
'''
from typing import Dict
from model import Embedding_layer,Scorer,SetinstanceClassifier
from dataloader import Dataloader
from pathlib import Path

class SynMineModelWrapper(object):
    """Class to wrapper training and testing of deeplearning model"""
    def __init__(self, modelconfig:Dict, trainingconfig:Dict) -> None:
        super(SynMineModelWrapper,self).__init__()
        embedding_layer = Embedding_layer.load(modelconfig['word2vec_path'])
        embedding_layer.freeze_parameters()
        hidden_size = modelconfig['hidden_size']
        assert len(hidden_size) == 2
        scorer = Scorer(embedding_layer,hidden_size[0],hidden_size[1])
        self.model = SetinstanceClassifier(scorer)

        self.epoches = trainingconfig['epoch']
        self.print_step = trainingconfig['print_step']
        self.init_lr = trainingconfig['lr']

        self.best_model = self.model
        self.best_loss = 1e10

        self.checkpoint_dir =trainingconfig['checkpoint_dir']



    def train(self,train_dataloader:Dataloader) -> None:
        for epoch in range(self.epoches):
            self.model.train()
            for item in train_dataloader:
                word_set, mask, new_word_set, mask_, labels = item    

        return 0
        

    def validate(self) -> None:
        return 0
