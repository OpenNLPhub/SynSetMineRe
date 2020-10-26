
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-20 15:58:45
@desc [description]
'''
from datetime import datetime
from logging import log
import torch
from torch import optim
from typing import Dict, None, Optional
from model import Embedding_layer,Scorer,SetinstanceClassifier
from dataloader import Dataloader, test_dataloader
from pathlib import Path
from log import logger
from tqdm import tqdm
from copy import deepcopy

class SynMineModelWrapper(object):
    """Class to wrapper training and testing of deeplearning model"""
    def __init__(self, modelconfig:Dict, trainingconfig:Dict) -> None:
        super(SynMineModelWrapper,self).__init__()
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device('cpu')
        
        embedding_layer = Embedding_layer.load(modelconfig['word2vec_path'])
        embedding_layer.freeze_parameters()
        hidden_size = modelconfig['hidden_size']
        assert len(hidden_size) == 2
        scorer = Scorer(embedding_layer,hidden_size[0],hidden_size[1])
        self.model = SetinstanceClassifier(scorer)
        self.loss_fn = modelconfig['loss_fn']
        self.start_epoches = 0
        self.epoches = trainingconfig['epoch']
        self.checkpoint_epoches = trainingconfig['checkpoint_epoch']
        self.print_step = trainingconfig['print_step']
        self.init_lr = trainingconfig['lr']

        self.best_model = self.model.to(self.device)
        self.best_loss = 1e10

        self.checkpoint_dir =trainingconfig['checkpoint_dir']

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, amsgrad=True)


    def train(self, train_dataloader:Dataloader, dev_dataloader:Optional[Dataloader] = None) -> None:
        all_step = len(train_dataloader)
        t = range(self.start_epoch, self.epoches)

        if dev_dataloader == None:
            '''split train_dataloader'''
            train_dataloader_, dev_dataloader_ = train_dataloader.split()
        else:
            train_dataloader_ = train_dataloader
            dev_dataloader_ = dev_dataloader
        # used to plot
        ep_loss_list = []
        val_loss_list = []

        for epoch in t:
            self.model.train()
            ep_loss = 0

            for step,item in enumerate(train_dataloader_):
                word_set, mask, new_word_set, mask_, labels = item
                F = lambda x: torch.Tensor(x).long().to(self.device)
                word_set_tensor, mask, new_word_set_tensor, mask_ = [
                    i for i in [word_set, mask, new_word_set, mask_]
                ]
                pred_labels = self.model(word_set_tensor, mask, new_word_set_tensor, mask_)
                #batch_size * 2 : vector
                labels = [[1,0] if i == 0 else [0,1] for i in labels]
                labels_tensor = torch.Tensor(labels).float().to(self.device)
                cur_loss = self.loss_fn(pred_labels,labels_tensor) / labels_tensor.shape[0]
                ep_loss += cur_loss.item()
                # backward
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()

                # log  
                if (step+1) % self.print_step == 0 or step + 1 == all_step:
                    desc = 'Epoch:{} {} / {} Loss:{}'.format(epoch,step,all_step,cur_loss.item())
                    logger.info(desc)

            #validate
            val_loss = self.validate(dev_dataloader_)
            #save checkpoint
            if epoch + 1 == self.start_epoches + self.checkpoint_epoches:
                #update best_model
                self.save_check_point()
                self.start_epoches = epoch

            val_loss_list.append(val_loss_list)
            ep_loss_list.append(ep_loss)
    def validate(self,dev_dataloader:Dataloader) -> None:
        #No dev dataset, so training dataset is validation dataset\
        self.model.eval()
        val_loss = 0
        for step,item in dev_dataloader:
            word_set, mask, new_word_set, mask_, labels = item
            F = lambda x: torch.Tensor(x).long().to(self.device)
            word_set_tensor, mask, new_word_set_tensor, mask_ = [
                i for i in [word_set, mask, new_word_set, mask_]
            ]
            labels_ = [[1,0] if i == 0 else [0,1] for i in labels]
            pred_labels = self.model(word_set_tensor, mask, new_word_set_tensor, mask_)
            labels_tensor = torch.Tensor(labels_).float().to(self.device)
            cur_loss = self.loss_fn(pred_labels,labels_tensor) / labels_tensor.shape[0]
            val_loss += cur_loss
            pred_labels = torch.argmax(pred_labels, dim=1)
            #metrics
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = deepcopy(self.model)

        #log
        return val_loss

        
            
    
    def test(self, test_dataloader:Dataloader) -> None:
        self.best_model.eval()
        self.best_model.to(self.device)
        all_step = len(test_dataloader)
        for step,item in enumerate(test_dataloader):
            word_set, mask, new_word_set, mask_, labels = item 
            F = lambda x: torch.Tensor(x).long().to(self.device)
            word_set_tensor, mask, new_word_set_tensor, mask_ = [
                i for i in [word_set, mask, new_word_set, mask_]
            ]
            pred_labels = self.best_model(word_set_tensor, mask, new_word_set_tensor, mask_)
            pred_labels = torch.argmax(pred_labels,dim=1)

            # cal metrics
            
            if (step+1) % self.print_step == 0 or (step+1) == all_step:
                desc = ''
                logger.info('Test:'+desc)
    
    def load_check_point(self,file_name:Optional[str] = None) -> None:
        dir_path = Path(self.checkpoint_dir)
        if file_name == None:
            import os
            flist = os.listdir(dir_path)
            filepath = Path.joinpath(dir_path,max(flist))
        else:
            filepath = Path.joinpath(dir_path,file_name)
        checkpoint = torch.load(filepath)
        self.best_loss = checkpoint['best_loss']
        self.start_epoches = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict']).to(self.device)
        self.best_model.load_state_dict(checkpoint['best_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def save_check_point(self) -> None:
        """
        d = {
            'epoch' : epoch,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_loss' : best_loss,
            'best_pred' : best_pred
        }
        """
        dir_path = Path(self.checkpoint_dir)
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filepath = Path.joinpath(dir_path,now+'_checkpoint.pth.tar')
        d = {
            'epoch':self.start_epoches,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_loss' : self.best_loss,
            'best_model' : self.best_model.state_dict()
        }
        torch.save(d, filepath)
        