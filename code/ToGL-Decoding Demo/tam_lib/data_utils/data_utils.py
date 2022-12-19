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

def read_and_process_data(source_df,
                          output_dir: str,
                          text_col: str = 'text',
                          val_frac: str = 0.1,
                          vect_min_df: Union[int, float] = 10,
                          vect_max_df: Union[int, float] = 0.7,
                          min_toks: int = 2,
                          max_vocab: int = None):
        
    dfs = {}
             
    print('Calculating length of input texts')
    source_df['len'] = source_df[text_col].str.split(' ').progress_apply(len)

    if val_frac > 0:
        dfs['train'], dfs['val'] = train_test_split(source_df, test_size = val_frac)
    else:
        dfs['train'] = source_df
    
    print('Sorting data by length')
    dfs = {
        split: df.sort_values('len', ascending = True)
        for split, df in dfs.items()
    }
    
    print(f'Cleaning text data')
    for split, df in dfs.items():
        df['text_clean'] = df[text_col].astype(str).progress_apply(lambda text: clean.kitchen_sink(text))
    
    
    print(f'Fitting vectorizer')
    vectorizer = CountVectorizer(min_df = vect_min_df, max_df = vect_max_df, max_features = max_vocab)
    vectorizer = vectorizer.fit(dfs['train']['text_clean'].astype(str))
    print(f'Vectorizer trained with vocab size {len(vectorizer.vocabulary_)}')        
    
    print(f'Filtering documents with less than {min_toks} tokens')
        
    bows = {
        split: vectorizer.transform(df['text_clean']) 
        for split, df in dfs.items()   
    }

    ix_keeps = {
        split: np.argwhere(np.asarray(bow.sum(axis=-1)).squeeze() >= min_toks).squeeze()
        for split, bow in bows.items()
    }

    dfs = {
        split: df.iloc[ix_keeps[split]]
        for split, df  in dfs.items()
    }
    
    dfs = {
        split: df.reset_index().rename({'index': 'source_id'}, axis = 1) 
        for split, df in dfs.items()
    }     
    
    
    print(f'Saving data to {output_dir}/{str(list(dfs.keys()))}.csv')
    
    if not os.path.exists(output_dir):
        print(f'{output_dir} does not exist, creating directory.')
        os.mkdir(output_dir)
        
    for split, df in dfs.items():
        df.to_csv(output_dir + f'/{split}.csv', index = None)
    
    with open(output_dir + '/vectorizer.pkl', 'wb') as f:
        pkl.dump(vectorizer, f)
    
    
    params = {
        'val_frac': val_frac,
        'vec_min_df': vect_min_df,
        'vec_max_df': vect_max_df,
        'min_toks': min_toks
    }
    
    with open(output_dir + '/params.json', 'w') as f:
        json.dump(params, f)
        
    print(f'Saved all data, vectorizer to {output_dir}')


def load_data(data_dir:str):
    """
        Load all data and vectorizer from a data directory
        
        Parameters:
            data_dir: str
                Directory containing data files and vectorizer pickle. Should be directory resulting from call to 'read_and_process_data'
                
        Return:
            train_data, test_data, val_data: pd.DataFrame
            vectorizer: CountVectorizer
    """
    
    print('Reading data files')
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df   = pd.read_csv(f'{data_dir}/val.csv')
    
    print('Reading vectorizer')
    with open(data_dir + '/vectorizer.pkl', 'rb') as f:
        vectorizer = pkl.load(f)
        
    
    train_df = train_df[~train_df['text_clean'].isna()]
    val_df   = val_df[~val_df['text_clean'].isna()]
    
    return train_df, val_df, vectorizer