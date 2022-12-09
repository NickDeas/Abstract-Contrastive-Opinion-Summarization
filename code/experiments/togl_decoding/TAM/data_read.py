import pandas as pd
import numpy as np
import pickle as pkl
import json
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from ctt import clean

from typing import Union, List

from tqdm import tqdm
tqdm.pandas()

def load_data(data_dir:str, embeds = False, verbs = False):
    """
        Load all data and vectorizer from a data directory
        
        Parameters:
            data_dir: str
                Directory containing data files and vectorizer pickle. Should be directory resulting from call to 'read_and_process_data'
                
        Return:
            train_data, test_data, val_data: pd.DataFrame
            vectorizer: CountVectorizer
    """
    if verbs:
        v_cols = ['verbs', 'nverbs']
        v_vecs = {}
        v_embeds = {}
    
    print('Reading data files')
    train_df, test_df, val_df = [
        pd.read_csv(f'{data_dir}/{split}.csv') for split
        in ['train', 'test', 'val']   
    ]
    
    print('Reading vectorizer')
    with open(data_dir + '/vectorizer.pkl', 'rb') as f:
        vectorizer = pkl.load(f)
    if embeds: 
        print('Reading word embeddings')
        with open(data_dir + '/embeds.npy', 'rb') as f:
            embeds = np.load(f)
            
        if verbs:
            print('Reading event-related vectorizers and embeddings')
            for v_col in v_cols:
                with open(data_dir + f'/{v_col}_vectorizer.pkl', 'rb') as f:
                    vec = pkl.load(f)
                    v_vecs[v_col] = vec
                
                with open(data_dir + f'/{v_col}_embeds.npy', 'rb') as f:
                    emb = np.load(f)
                    v_embeds[v_col] = emb
            
            return train_df, test_df, val_df, vectorizer, embeds, v_vecs, v_embeds
        else:
            return train_df, test_df, val_df, vectorizer, embeds
    else:
        if events:
            print('Reading event-related vectorizers')
            for v_col in v_cols:
                with open(data_dir + f'/{v_col}_vectorizer.pkl', 'rb') as f:
                    vec = pkl.load(f)
                    v_vecs[v_col] = vec
                
        else:
            return train_df, test_df, val_df, vectorizer, v_vecs
    
    return train_df, test_df, val_df, vectorizer