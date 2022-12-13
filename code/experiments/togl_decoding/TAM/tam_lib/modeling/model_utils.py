from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

def train_epoch(topic_model, svi, train_dl, progress_interval = -1, use_author = False):
    """
    Parameters:
        topic_model: nn.Module
            Topic model implemented in pyro and trainable via SVI (ex. ProdLDA or ProdHLDA)
        svi: 
            Instance of pyro's Stochastic Variational Inference (SVI) for training
        train_dl: DataLoader
            Torch dataloader with batches of training data
        progress_interval: int (optional, default = -1)
            Number of steps between progress information is updated. Value of -1 ignores prints
    """
    
    topic_model.train()
    device = topic_model.device
    
    # Initialize loss accumulator
    running_loss = 0.
    total_num_words = 0
    
    # Iterate over training batches for one epoch
    for i, data in enumerate(train_dl): 
        bow = data['bow'].to(device)   
        
        if use_author:
            ids = data['author_id'].to(device)

            running_loss += svi.step(bow, ids, num_docs = len(train_dl.dataset))
        else:
            running_loss += svi.step(bow, num_docs = len(train_dl.dataset))
        total_num_words += bow.sum().cpu().item()
        
        if progress_interval > 0 and i % progress_interval == 0:
            print(f"Step {i}; avg. loss {running_loss/total_num_words}", end='\r')

    # return epoch loss
    total_epoch_loss_train = running_loss / total_num_words
    return total_epoch_loss_train

def evaluate(topic_model: nn.Module,
             svi: SVI, 
             val_dl: DataLoader,
             use_author: bool = False,
             elbo_scale_factor: float = None): 
    """
    Evaluate a topic model on a holdout set
    
    Parameters:
        topic_model: nn.Module
            Topic model implemented in pyro and trainable via SVI (ex. ProdLDA or ProdHLDA)
        svi: 
            Instance of pyro's Stochastic Variational Inference (SVI) for training
        val_dl: DataLoader
            Torch dataloader with batches of evaluation data
        elbo_scale_factor: float (optional, default = None)
            Scaling to use to adjust for the fact that
            the elbo scale was set by the size of the training dataset
            most likely this factor should be set to `N_train / N_val`. 
    """
    
    topic_model.eval()
    device = topic_model.device
    
    # Initialize loss accumulator and word count
    total_loss = 0.
    total_num_words = 0
    
    # Compute loss over the entire test set
    for data in val_dl:            
        bow = data['bow'].to(device)
        total_loss += svi.evaluate_loss(bow, num_docs = len(val_dl.dataset))
            
        total_num_words += bow.sum().cpu().item()
        
    epoch_elbo = total_loss / total_num_words
    
    if elbo_scale_factor is not None:
        epoch_elbo *= elbo_scale_factor
    
    return epoch_elbo



def train(topic_model: nn.Module, 
          svi: SVI, 
          train_dl: DataLoader,
          num_epochs: int,
          val_dl: DataLoader = None,
          test_frequency: int = 1,
          progress_interval = -1,
          use_author:bool = False,
          save_loc: str = './temp_model',
          stall_stop: int = 5):
    """
    Train a topic model for a given number of epochs
    Parameters:
        topic_model: nn.Module
            Topic model implemented in pyro and trainable via SVI (ex. ProdLDA or ProdHLDA)
        svi: 
            Instance of pyro's Stochastic Variational Inference (SVI) for training
        train_dl: DataLoader
            Torch dataloader with batches of training data
        num_epochs: int
            Number of epochs/iterations through dataset to train for
        val_dl: DataLoader
            Torch dataloader with batches of evaluation data
        test_frequency: int
            Interval between epochs to evaluate the topic model. Only performed if val_dl is passed
        progress_interval: int (optional, default = -1)
            Number of steps between progress information is updated. Value of -1 ignores prints
    """
    
    # Initialize empty ELBO logs
    train_elbo = []
    val_elbo = []
    min_val_elbo = np.inf
    stall_count = 0

    print("Beginning Training")
    for epoch in range(num_epochs):
        if stall_count > stall_stop:
            break
            
        # Training
        total_epoch_loss_train = train_epoch(topic_model, svi, train_dl, use_author = use_author)
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  Training ELBO:    %.4f" % (epoch, total_epoch_loss_train))

        # Validation
        if test_frequency > 0 and val_dl and epoch % test_frequency == 0:
            total_epoch_loss_val = evaluate(topic_model, svi, val_dl, 
                                            elbo_scale_factor=len(train_dl.dataset) / len(val_dl.dataset),
                                            use_author = use_author)
            val_elbo.append(total_epoch_loss_val)

            print(" " * 13 + "Validation ELBO:  %.4f" % (total_epoch_loss_val))
            
            if total_epoch_loss_val < min_val_elbo:
                min_val_elbo = total_epoch_loss_val
                stall_count = 0
                topic_model.save(save_loc)
            else:
                stall_count += 1
    
    return train_elbo, val_elbo