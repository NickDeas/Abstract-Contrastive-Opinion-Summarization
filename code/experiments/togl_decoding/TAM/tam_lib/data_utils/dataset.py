import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer

from typing import Union, List

import re



class DocDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 text_col: str = 'text_clean', 
                 vectorizer: CountVectorizer = None,
                 min_n2: int = 3):
        """
        Parameters:
                -df: dataframe
                -text_col: the name of the column in df containing the textual data
                -author_id_col: the name of the column in df containing the author ID data
                -vectorizer: the vectorizer to use in transforming text into BOW
                -split_halves: can be one of ('ordered', 'random', or None). Ordered returns first and second halves of the text, 
                    random returns randomly selected halves, and None ignores the 'h1' and 'h2' indices
                
        """
        self.data = df
        self.text_col = text_col

        # get vectorized text
        self.vectorizer = vectorizer if vectorizer else CountVectorizer()
        self.bows = self.vectorizer.transform(self.data[text_col].astype(str)).astype(np.float32)         
        
        def _filter_vocab_text(text, vocab):
            return ' '.join([term for term in self.vectorizer.build_tokenizer()(text) if self.vectorizer.build_preprocessor()(term) in vocab.keys()])
        
        self.data[self.text_col + '_filt'] = self.data[self.text_col].apply(lambda text: _filter_vocab_text(text, self.vectorizer.vocabulary_))
            
    
    def __len__(self):
        """Return number of samples in the dataset"""
        return self.data.shape[0]

    def __getitem__(self, ix):
        """Returns a dictionary from the ith unique author containing the following:
            -author ID
            -BOW document tensor of size [[self.samples, vocab size]] (not normalized)
                This may contain repeated documents if an author has less documents
                than the desired number of samples per author
        """
        bow = self.bows[ix].toarray().astype(np.float32)
        
        return_dict = {'bow': bow, 'title': self.data['title'].values[ix]}
        
        return return_dict
    
    def max_len(self):
        return self.bows.sum(-1).max()