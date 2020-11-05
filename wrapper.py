
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-20 15:58:45
@desc [description]
'''
from datetime import datetime
import enum

from numpy.core.defchararray import index
from utils import set_padding
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Dict, Optional, Sequence
import numpy as np
from dataloader import DataSet, Dataloader
from pathlib import Path
from log import logger
from copy import deepcopy
from evaluate import EvalUnit,binary_confusion_matrix_evaluate

class ModelWrapper(object):
    """Class to wrapper training and testing of deeplearning model
    
    """
    def __init__(self, model:nn.Module, trainingconfig:Dict) -> None:
        super(ModelWrapper,self).__init__()
        self.device = torch.device(trainingconfig['cuda']) if torch.cuda.is_available() else torch.device('cpu')
        
        # embedding_layer = Embedding_layer.load(modelconfig['word2vec_path'])
        # embedding_layer.freeze_parameters()
        # hidden_size = modelconfig['hidden_size']
        # assert len(hidden_size) == 2
        # scorer = Scorer(embedding_layer,hidden_size[0],hidden_size[1])
        self.model = model
        self.loss_fn = nn.BCELoss()

        self.start_epoches = 0
        self.threshold = trainingconfig['threshold']
        self.epoches = trainingconfig['epoch']
        self.checkpoint_epoches = trainingconfig['checkpoint_epoch']
        self.print_step = trainingconfig['print_step']
        self.init_lr = trainingconfig['lr']

        self.best_model = self.model.to(self.device)
        self.best_loss = 1e10
        self.best_score = 0
        self.checkpoint_dir =trainingconfig['checkpoint_dir']
        self.batch_size = trainingconfig['batch_size']
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.init_lr, amsgrad=True)
        # optimizer is default 

    def train(self, train_dataloader:Dataloader, dev_dataloader:Optional[Dataloader] = None) -> None:
        """Implementation to Batch Train the model
        Args:
            train_dataloader : training dataset iteration used to train model
            dev_dataloader : development dataset iteration used to validate model
                if dev_dataloader = None, it will split the trianing dataset to create dev dataset
        Returns:
            None
        """
        writer = SummaryWriter()

        all_step = len(train_dataloader)
        t = range(self.start_epoches, self.epoches)

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
            summary_metrics = EvalUnit(0,0,0,0,'epoch'+str(epoch))
            for step,item in enumerate(train_dataloader_):
                word_set, mask, new_word_set, mask_, labels = item
                F = lambda x: torch.Tensor(x).long().to(self.device)
                word_set_tensor, mask, new_word_set_tensor, mask_ = [
                    F(i) for i in [word_set, mask, new_word_set, mask_]
                ]
                pred_labels = self.model(word_set_tensor, mask, new_word_set_tensor, mask_)
                #batch_size * 1 : vector
                labels_tensor = torch.Tensor(labels).float().to(self.device)
                cur_loss = self.loss_fn(pred_labels,labels_tensor) / labels_tensor.shape[0]
                ep_loss += cur_loss.item()
                # backward
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()
                # cal metrics
                pred_labels = np.where(pred_labels.cpu().detach().numpy()>self.threshold, 1, 0)
                # logger.info('{}'.format(step))
                # if step == 20:
                #     import pdb;pdb.set_trace()
                unit = binary_confusion_matrix_evaluate(np.array(labels),pred_labels)
                summary_metrics += unit
                # log  
                if (step+1) % self.print_step == 0 or step + 1 == all_step:
                    desc = 'Epoch:{} {} / {} Loss:{}'.format(epoch+1,step+1,all_step,cur_loss.item())
                    logger.info(desc)
            #log metrics
            logger.info("In Training Set, Metrics:")
            logger.info(summary_metrics)

            #plot
            # writer.add_scalar('Loss/Train',)

            #validate
            val_loss = self.validate(dev_dataloader_)

            #save checkpoint
            if (epoch + 1 == self.start_epoches + self.checkpoint_epoches or 
                epoch + 1 == self.epoches):
                #update best_model
                self.save_check_point()
                self.start_epoches = epoch
                logger.info('Successfully Store Checkpoint in epoch {}'.format(epoch+1))

            val_loss_list.append(val_loss)
            ep_loss_list.append(ep_loss / all_step)
        # tensorboard plot
        writer = SummaryWriter()
        for i,loss in enumerate(ep_loss_list):
            writer.add_scalar('Loss/Train', loss, i)
        for i, loss in enumerate(val_loss_list):
            writer.add_scalar('Loss/Validation', loss, i)
        writer.close()
        
    def validate(self,dev_dataloader:Dataloader) -> Any:
        """Implementation to Batch validate the model using developed dataset
        Args:
            dev_dataloader : development dataset iteration used to validate model
        """
        self.model.eval()
        val_loss = 0
        summary_metrics = EvalUnit(name='Validation')
        all_step = len(dev_dataloader)
        for step,item in enumerate(dev_dataloader):
            word_set, mask, new_word_set, mask_, labels = item
            F = lambda x: torch.Tensor(x).long().to(self.device)
            word_set_tensor, mask, new_word_set_tensor, mask_ = [
                F(i) for i in [word_set, mask, new_word_set, mask_]
            ]
            pred_labels = self.model(word_set_tensor, mask, new_word_set_tensor, mask_)
            labels_tensor = torch.Tensor(labels).float().to(self.device)
            cur_loss = self.loss_fn(pred_labels,labels_tensor) / labels_tensor.shape[0]
            val_loss += cur_loss.item()
            #metrics
            pred_labels = np.where(pred_labels.cpu().detach().numpy()>self.threshold, 1, 0)
            unit = binary_confusion_matrix_evaluate(np.array(labels),pred_labels)
            summary_metrics += unit
            if (step+1) % self.print_step == 0 or step + 1 == all_step:
                logger.info("Validation: step / all_step : {} / {}, loss:{}".format(step+1,all_step,cur_loss.item()))

        #log
        desc = 'In Validation, Average Loss:{:.2f}'.format(val_loss / all_step)
        logger.info(desc)
        logger.info(summary_metrics)

        if summary_metrics.f1_score() >= self.best_score:
            self.best_loss = val_loss
            self.best_score = summary_metrics.f1_score()
            self.best_model = deepcopy(self.model)
            logger.info('Successfully Update Best Model')
        
        return val_loss / all_step
   
    def test(self, test_dataloader:Dataloader) -> None:
        """Implementation to Batch test the model using developed dataset
        Args:
            test_dataloader : test dataset iteration used to test the model
        """
        self.best_model.eval()
        self.best_model.to(self.device)
        all_step = len(test_dataloader)
        summary_metrics = EvalUnit()
        for step,item in enumerate(test_dataloader):
            word_set, mask, new_word_set, mask_, labels = item 
            F = lambda x: torch.Tensor(x).long().to(self.device)
            word_set_tensor, mask, new_word_set_tensor, mask_ = [
                F(i) for i in [word_set, mask, new_word_set, mask_]
            ]
            pred_labels = self.best_model(word_set_tensor, mask, new_word_set_tensor, mask_)
            pred_labels = np.where(pred_labels.cpu().detach().numpy()>self.threshold, 1, 0)
            # cal metrics
            unit = binary_confusion_matrix_evaluate(np.array(labels),pred_labels)
            summary_metrics += unit
            logger.info("Test: step / all_step : {} / {}".format(step+1,all_step))

        logger.info("In Test DataSet")
        logger.info(summary_metrics)
    
    def load_check_point(self,file_name:Optional[str] = None) -> None:
        """load the save model file
        Args:
            file_name: filename not filepath, it will concatenate the checkpoint_dir and filename 
                to create the whole path. It filename is none, it will choose the latest checkpoint file
                under the checkpoint directory.
        """
        dir_path = Path(self.checkpoint_dir)
        if file_name == None:
            import os
            flist = os.listdir(dir_path)
            if not flist:
                logger.info('No checkpoint file')
                raise ValueError()
            filepath = Path.joinpath(dir_path,max(flist))
        else:
            filepath = Path.joinpath(dir_path,file_name)
        checkpoint = torch.load(filepath, map_location='cpu')
        self.best_loss = checkpoint['best_loss']
        self.start_epoches = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.best_model.load_state_dict(checkpoint['best_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_score = checkpoint['best_score']
    
    def save_check_point(self) -> None:
        """ save the key attribute in this wrapper
        key attribute:
        d = {
            'epoch' : epoch,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_loss' : best_loss,
            'best_score' : best_score,
            'best_model' : best_pred
        }
        """
        dir_path = Path(self.checkpoint_dir)
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filepath = Path.joinpath(dir_path,self.model.name+'_'+self.model.version+'_'+now+'_checkpoint.pth.tar')
        d = {
            'epoch':self.start_epoches,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_loss' : self.best_loss,
            'best_model' : self.best_model.state_dict(),
            'best_score' : self.best_score
        }
        torch.save(d, filepath)
    
    def cluster_predict(self, dataset:DataSet, word2id:Dict, outputfile:Optional[Path]) -> Sequence[Any]:
        """Using Model to cluster wordset
        Args:
            dataset: it's self defined class, in DataSet, we use vocab to get all words and true cluster result
            word2id: it is got from embedding file, translate word to embedding index
            outputfile: outputfile path
        Returns:
            List of word sets
        """
        self.best_model.eval()
        vocab = dataset.vocab
        words = vocab.keys()
        wordset_list = []
        # import pdb;pdb.set_trace()
        for word in words:
            wordid = word2id[word]
            if not wordset_list:
                # Empty
                wordset_list.append([wordid])
                continue
            new_wordset = [[*i, wordid] for i in wordset_list]
            old_wordset = deepcopy(wordset_list)
            itemsnum = len(new_wordset)
            # add batch operation
            tmp_best_scores = 0
            index = 0
            for ix in range(0,itemsnum,self.batch_size):
                batch_new_wordset = new_wordset[ix:ix+self.batch_size]
                batch_old_wordset = old_wordset[ix:ix+self.batch_size]
                batch_old_wordset, mask = set_padding(batch_old_wordset)
                batch_new_wordset, mask_ = set_padding(batch_new_wordset)
                #batch_size * max_word_set_size
                F = lambda x: torch.Tensor(x).long().to(self.device)
                word_set_tensor, mask, new_word_set_tensor, mask_ = [
                    F(i) for i in [batch_old_wordset, mask, batch_new_wordset, mask_]
                ]
                scores = self.best_model(word_set_tensor, mask, new_word_set_tensor, mask_)
                best_scores = torch.max(scores).item()
                if best_scores >= tmp_best_scores:
                    tmp_best_scores = best_scores
                    index = ix + torch.argmax(scores).item()

            if tmp_best_scores > self.threshold:
                wordset_list[index].append(wordid)
            else:
                wordset_list.append([wordid])
        #id2word
        # import pdb;pdb.set_trace()
        id2word = { j:i for i,j in word2id.items()}
        F = lambda x:[ id2word[i] for i in x]
        pred_word_sets = [ F(wordset) for wordset in wordset_list]
        
        if outputfile is not None:
            with open(outputfile, 'w', encoding='utf-8') as f:
                for pred_word_set in pred_word_sets:
                    for word in pred_word_set:
                        f.write(word+' ')
                    f.write('\n')

        return pred_word_sets

    def evaluate(self, dataset:DataSet, pred_word_sets:Sequence[Any], function_list:Sequence[Callable[...,float]])->Sequence[Any]:
        """ Use Evaluating Function to Evaluate the final result
        Args:
            dataset: it's self defined class, we use vocab attribute to get true cluster result
            pred_word_set: the output of cluster_predict method | List of word sets
            function_list: the list of evaluating function which have two input pred_cluster and target_cluster
        """

        #trans datatype 
        clusters = set(dataset.vocab.values())
        cluster2id = {cluster:idx for idx,cluster in enumerate(clusters)}
        target_cluster = {key:cluster2id[value] for key,value in dataset.vocab.items()}
        pred_cluster = {}
        # import pdb;pdb.set_trace()
        for idx,pred_word_set in enumerate(pred_word_sets):
            for word in pred_word_set:
                pred_cluster[word] = idx
        # import pdb;pdb.set_trace()
        ans = []
        for func in function_list:
            ans.append(func(pred_cluster = pred_cluster,target_cluster = target_cluster))
        return ans
