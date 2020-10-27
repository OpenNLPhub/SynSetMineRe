
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-19 16:29:05
@desc [description]
'''

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.linear import Linear



class Embedding_layer(nn.Module):
    """embedding layer and provide some method to freeze layer parameters"""
    def __init__(self, vocab_size, embedding_dim)->None:
        super(Embedding_layer,self).__init__()
        self.dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.dim, padding_idx = 0)
        # self._freeze_parameters()

    @classmethod
    def load(cls,word_file):
        """Polymorphism Contructor """
        word2vec = torch.from_numpy(np.load(word_file))
        vocab_size, embedding_dim = word2vec.shape
        layer = cls(vocab_size,embedding_dim)
        layer.embedding = nn.Embedding.from_pretrained(word2vec, padding_idx=0).float()
        return layer

    def forward(self,x):
        """
        Args:
            x:  batch_size, max_word_set_size
        Returns:
            word embedding value 
            batch_size, max_word_set_size, word_emb_size 
        """
        return self.embedding(x)


    def freeze_parameters(self):
        for i in self.embedding.parameters():
            i.requires_grad = False

    def unfreeze_parameters(self):
        for i in self.embedding.parameters():
            i.requires_grad = True

# def getScorer():


class Scorer(nn.Module):
    """Module to score word set"""
    def __init__(self,embedding_layer:Embedding_layer, embed_trans_hidden_size:int, post_trans_hidden_size:int, dropout=0.1) -> None:
        """
        Args:
            embedding_layer: word2vec module
            embed_trans_hidden_size: tuple of hidden size in embedding_transformer module. According to paper, it contains 2 elements 
            post_trans_hidden_size: tuple of hidden size in post_transformer module. According to paper, it contains 3 elements
        """
        super(Scorer,self).__init__()
        self.embed_trans_hidden_size = embed_trans_hidden_size
        self.post_trans_hidden_size = post_trans_hidden_size
        self.embedding_layer = embedding_layer
        self.embedding_transformer = nn.Sequential(
            nn.Linear(self.embedding_layer.dim,self.embedding_layer.dim, bias = False),
            nn.ReLU(),
            nn.Linear(self.embedding_layer.dim,self.embed_trans_hidden_size),
            nn.ReLU()
        )
        self.post_transformer = nn.Sequential(
            nn.Linear(self.embed_trans_hidden_size,self.post_trans_hidden_size),
            nn.ReLU(),
            nn.Linear(self.post_trans_hidden_size,self.post_trans_hidden_size / 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.post_trans_hidden_size / 2, 1)
        )
    
    def forward(self, input_ids, mask):
        """
        Args:
            x: [input_ids, mask]
            input_ids:  a batch of word sets | batch_size, max_set_size
            mask: notification of padding
        """

        y = self.embedding_layer(input_ids)
        # batch_size, max_set_size, embedding_dim 

        y = self.embedding_transformer(y)
        # batch_size, max_set_szie, embed_trans_hidden_size

        #sum
        mask_ = mask.unsqueeze(-1).expand(-1,-1,y.shape[-1])
        y = y*mask_
        y = torch.sum(y,dim = 1)
        # batch_size, embed_trans_hidden_size

        y = self.post_transformer(y)
        # batch_size, 1
        y = y.squeeze(-1)
        # y = torch.softmax(y,dim=1)
        return y


class SetinstanceClassifier(nn.Module):
    """Classifier to predict waiting word is or not in input set"""
    def __init__(self,scorer:Scorer)->None:
        super(SetinstanceClassifier,self).__init__()
        self.scorer = scorer

    def forward(self,word_set, mask, new_word_set, new_mask):
        """
        Args:
            word_set: batch_size * max_word_size
            mask: word_set mask
            new_word_set: batch_size * max_word_size
            new_mask: new word set mask 
        """
        old_score = self.scorer(word_set,mask)
        new_score = self.scorer(new_word_set,new_mask)
        # batch_size
        ans = torch.sigmoid(new_score - old_score)
        return ans
    