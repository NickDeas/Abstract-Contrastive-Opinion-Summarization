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
                 split_halves: bool =None,
                 split_frac: Union[int, float] = 0.5,
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
        self.split_halves = split_halves

        # get vectorized text
        self.vectorizer = vectorizer if vectorizer else CountVectorizer()
        self.bows = self.vectorizer.transform(self.data[text_col].astype(str)).astype(np.float32)
        
        # Filter data depending on splitting method (num tokens or fraction)
        if isinstance(split_frac, int) or split_frac >= 1:
            split_frac = int(split_frac)
            ix_keep = np.argwhere(self.bows.toarray().sum(axis=-1).squeeze() >= (split_frac + min_n2)).squeeze()
            self.bows = self.bows[ix_keep]
            self.data = self.data.iloc[ix_keep].reset_index()
            
            # raise ValueError('Debug')
            
        
        def _filter_vocab_text(text, vocab):
            return ' '.join([term for term in self.vectorizer.build_tokenizer()(text) if self.vectorizer.build_preprocessor()(term) in vocab.keys()])
        
        self.data[self.text_col + '_filt'] = self.data[self.text_col].apply(lambda text: _filter_vocab_text(text, self.vectorizer.vocabulary_))
        
        # split documents for perplexity computation
        if self.split_halves in ['ordered', 'random']:
            src_text_col = self.text_col + '_filt'
            
            if self.split_halves == 'random':
                # If random, shuffle the words before getting halves
                # This conserves duplicated words in the sentence and is more memory efficient to compute before vectorization
                self.data['shuffle_text'] = self.data[src_text_col].apply(lambda text: ' '.join(np.random.permutation(text.split(' '))))
                src_text_col = 'shuffle_text'
                                
            if isinstance(split_frac, int) or split_frac >= 1:
                self.data['text_h1'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[:split_frac]) )
                self.data['text_h2'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[split_frac:]) )
            else:
                self.data['text_h1'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[:int(len(text.split(' '))*split_frac)]) )
                self.data['text_h2'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[int(len(text.split(' '))*split_frac):]) )
        
            self.bows_h1 = self.vectorizer.transform(self.data['text_h1'].astype(str)).astype(np.float32)
            self.bows_h2 = self.vectorizer.transform(self.data['text_h2'].astype(str)).astype(np.float32)
            
    
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
        bow = self.bows[ix].toarray().astype(np.float32)[0]
        
        return_dict = {'bow': bow}
        
        if self.split_halves in ('ordered', 'random'):
            bow_h1 = self.bows_h1[ix].toarray().astype(np.float32)[0]
            bow_h2 = self.bows_h2[ix].toarray().astype(np.float32)[0]
            return_dict.update({'bow_h1': bow_h1, 'bow_h2': bow_h2})
        
        return return_dict
    
    def max_len(self):
        return self.bows.sum(-1).max()
    
    
class DocVerbDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 verb_col: str = 'verbs_clean',
                 nverb_col: str = 'nverbs_clean',
                 v_vectorizer: CountVectorizer = None, 
                 nv_vectorizer: CountVectorizer = None,
                 split_halves: bool = False,
                 split_frac: Union[int, float] = 0.5,
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
        self.data = df.fillna('')
        self.verb_col = verb_col
        self.nverb_col = nverb_col
        
        self.split_halves = split_halves

        # get vectorized text
        self.v_vectorizer = v_vectorizer if v_vectorizer else CountVectorizer()
        self.nv_vectorizer = nv_vectorizer if nv_vectorizer else CountVectorizer()
        self.v_bows = self.v_vectorizer.transform(self.data[verb_col].astype(str)).astype(np.float32)
        self.nv_bows = self.nv_vectorizer.transform(self.data[nverb_col].astype(str)).astype(np.float32)
        
        # Filter data depending on splitting method (num tokens or fraction)
        if isinstance(split_frac, int) or split_frac >= 1:
            split_frac = int(split_frac)
            ix_keep = np.argwhere(self.bows.toarray().sum(axis=-1).squeeze() >= (split_frac + min_n2)).squeeze()
            self.bows = self.bows[ix_keep]
            self.data = self.data.iloc[ix_keep].reset_index()
            
            # raise ValueError('Debug')
            
        
        def _filter_vocab_text(text, vocab, vectorizer):
            return ' '.join([term for term in vectorizer.build_tokenizer()(text) if vectorizer.build_preprocessor()(term) in vocab.keys()])
        
        self.data[self.verb_col + '_filt'] = self.data[self.verb_col].apply(lambda text: _filter_vocab_text(text, self.v_vectorizer.vocabulary_, self.v_vectorizer))
        self.data[self.nverb_col + '_filt'] = self.data[self.verb_col].apply(lambda text: _filter_vocab_text(text, self.nv_vectorizer.vocabulary_, self.nv_vectorizer))
        
#         # split documents for perplexity computation
#         if self.split_halves in ['ordered', 'random']:
#             src_text_col = self.text_col + '_filt'
            
#             if self.split_halves == 'random':
#                 # If random, shuffle the words before getting halves
#                 # This conserves duplicated words in the sentence and is more memory efficient to compute before vectorization
#                 self.data['shuffle_text'] = self.data[src_text_col].apply(lambda text: ' '.join(np.random.permutation(text.split(' '))))
#                 src_text_col = 'shuffle_text'
                                
#             if isinstance(split_frac, int) or split_frac >= 1:
#                 self.data['text_h1'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[:split_frac]) )
#                 self.data['text_h2'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[split_frac:]) )
#             else:
#                 self.data['text_h1'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[:int(len(text.split(' '))*split_frac)]) )
#                 self.data['text_h2'] = self.data[src_text_col].apply(lambda text: ' '.join(text.split(' ')[int(len(text.split(' '))*split_frac):]) )
        
#             self.bows_h1 = self.vectorizer.transform(self.data['text_h1'].astype(str)).astype(np.float32)
#             self.bows_h2 = self.vectorizer.transform(self.data['text_h2'].astype(str)).astype(np.float32)
            
    
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
        v_bow = self.v_bows[ix].toarray().astype(np.float32)[0]
        nv_bow = self.nv_bows[ix].toarray().astype(np.float32)[0]
        
        return_dict = {'v_bow': v_bow, 'nv_bow': nv_bow}
        
        # if self.split_halves in ('ordered', 'random'):
        #     bow_h1 = self.bows_h1[ix].toarray().astype(np.float32)[0]
        #     bow_h2 = self.bows_h2[ix].toarray().astype(np.float32)[0]
        #     return_dict.update({'bow_h1': bow_h1, 'bow_h2': bow_h2})
        
        return return_dict