import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances

from typing import List, Tuple, Union, Optional

class CMOS():
    '''
        Reimplementation of Contrastive Max-Sum Opinion Summarization
    '''
    
    def __init__(self, lambda_w: float, vectorizer: Union[CountVectorizer,TfidfVectorizer]):
        '''
            Constructor for CMOS module
            
            Parameters:
                -lambda_w: float
                    Weight of contrastiveness in summary predictions
                -vectorizer: 
                    CountVectorizer or TFIDFVectorizer for text representations
        '''
        
        self.lambda_w = lambda_w
        self.vectorizer = vectorizer
        self.SENT_SPLIT_TOK = '\|\|\|'
    
    def predict(self, l_text, r_text):
        '''
            Predict the representative, contrastive summaries using Contrastive Max-Sum Opinion Summarization.
            Text should not contain adjectives or negations
            
            Parameters:
                -l_text: str
                    "Left" source texts joined by '|||'
                -r_text: str
                    "Right" source texts joined by '|||'
        '''
        
        # Split the joined texts into lists of sentences
        half1 = self.split_text(l_text) if isinstance(l_text, str) else l_text
        half2 = self.split_text(r_text) if isinstance(r_text, str) else r_text
        
        # Generate BOW or TFIDF Representations of texts
        half1_bows = self.get_vecs(half1)
        half2_bows = self.get_vecs(half2)
        
        # Calculate cosine similarities
        half1_cs = self.calc_cos_sims(half1_bows)
        half2_cs = self.calc_cos_sims(half2_bows)
        
        # Combine cosine similarities into a single matrix for vectorized calculations
        sum_cs   = self.combine_cos_sims(half1_cs, half2_cs)
        
        # Calculate cosine distances
        dist_cs  = self.calc_cos_dists(half1_bows, half2_bows)
        
        # Calculate prediction scores using the CMOS formula
        all_scores    = sum_cs + 2 * self.lambda_w * dist_cs
        
        # Select the pair of sentences that maximize the prediction score
        h1_idx, h2_idx = np.unravel_index(all_scores.argmax(), all_scores.shape)
        
        return half1[h1_idx], half2[h2_idx]
    
    def calc_cos_sims(self, bows):
        '''
            Calculate cosine similarities of each sentence to all others
            
            Parameters:
                -bows
                    BOW or TFIDF representations of sentences
            
            Return
                The mean of each sentences cosine similarity to all other sentences in the group
        '''
        
        sims = 1. - pairwise_distances(bows, metric = 'cosine')
        sims = sims.mean(-1)
        return sims
    
    def combine_cos_sims(self, cs1s, cs2s):
        '''
            Combine cosine similarities into a single matrix for vectorized calculations
            
            Parameters:
                cs1s:
                    Cosine similarities of one set of source documents
                cs2s:
                    Cosine similarities of the other set of source documents
            
            Return
                A matrix of len(cs1s) x len(cs2s) holding the sum of cosine similarity scores cs1s[i] and cs2s[j]
        '''
        
        combos = itertools.product(cs1s, cs2s)
        combos = [a + b for (a, b) in combos]
        combos = np.array(combos).reshape(len(cs1s), len(cs2s))
        return combos
    
    def calc_cos_dists(self, bows1, bows2):
        '''
            Calculate cosine distances between two sets of documents
            
            Parameters:
                -bows1
                    BOW or TFIDF representations of one set of documents
                -bows2
                    BOW or TFIDF representations of the other set of documents
                    
            Return
                Matrix where M_i,j holds the cosine distance between bows1[i] and bows2[j]
        '''
        
        dists = pairwise_distances(bows1, bows2, metric = 'cosine')
        return dists
    
    def get_vecs(self, text_list):
        '''
            Get BOW/TFIDF representations of a list of texts
            
            Parameters:
                -text_list:
                    List of texts to encode
            
            Return
                Matrix of vectorizer representations of the given texts
        '''
        return self.vectorizer.transform(text_list)
    
    def split_text(self, text):
        '''
            Split a text into its sentences
            
            Parameters:
                -text
                    Text to split
            
            Return
                Original text split by '|||' into sentences
        '''
        
        text_l = re.split(self.SENT_SPLIT_TOK, text)
        text_l = [t for t in text_l if t != '']
        return text_l