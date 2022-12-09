import numpy as np
from scipy import sparse
import math
import statistics
import torch
import torch.nn.functional as F

from typing import List, Union, Tuple, Callable

EPS = 1e-6 # Constant used in calculation of log terms in topic coherence

# ----- TOPIC COHERENCE -----

def calc_freqs(bows_train: sparse.csr.csr_matrix, 
               topic_ixs: List[List[int]]
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Vectorized calculation of individual and joint term frequencies for all terms in a single topic
        
        Arguments:
            -bows_train: sparse.csr.csr_matrix
                Bag of words representations of all documents in the source corpus
            -topic_ixs: List[List[int]]
                Word id's for all terms in single topic
        
        Return: Tuple of (topic_terms, term_freqs, joint_freqs)
            topic_terms: np.ndarray
                List of topic term ids (not guaranteed to be in the same order 
                    as passed),
            term_freqs: np.ndarray
                Individual term frequencies for all passed terms in topic 
                    (aligned with first return parameter),
            joint_freqs: np.ndarray
                Joint frequency matrix where A[i][j] is the joint frequency of 
                    topic_terms[i] and topic_terms[j]
    """
    
    topic_ixs = np.array(topic_ixs)
    topic_terms = np.unique(topic_ixs)
    
    
    bows_train_topic = bows_train[:, topic_terms].toarray()
        
    bows_1 = bows_train_topic[...,None]
    bows_2 = bows_train_topic[:, None]
    
    term_freqs = (bows_train_topic > 0).sum(axis = 0).squeeze() / bows_train_topic.shape[0]
    
    joint_freqs = (bows_1>=1) & (bows_2>=1)
    joint_freqs = joint_freqs.sum(axis = 0)/bows_train_topic.shape[0]
    
    
    return topic_terms, term_freqs, joint_freqs

def single_topic_coh(bows_train: sparse.csr.csr_matrix, 
                     topic_ixs: List[List[int]]) -> float:
    """
        Calculate the topic coherence for a single topic or single list of terms
        
        Parameters:
            -bows_train: sparse.csr_matrix
                Bag of words representations of all documents in the source corpus
            -topic_ixs: List[List[int]]
                Word id's for all terms in single topic
        
        Return: float
            The calculated topic coherence for the given topic
    """
    
    topic_coh = 0
    topic_terms, term_freqs, joint_freqs = calc_freqs(bows_train, topic_ixs)
    
    # Iterate over terms in the topic
    for i, term1 in enumerate(topic_terms[:-1]):
        # Iterate over remaining terms in the topic
        for j, term2 in enumerate(topic_terms[i+1:]):
            k = j + i + 1
            term1_freq = term_freqs[i]
            term2_freq = term_freqs[k]
            joint_freq = joint_freqs[i][k]
            
            topic_coh += ( np.log((joint_freq + EPS)/(term1_freq * term2_freq))) / (-1*np.log(joint_freq + EPS))
    
    num_pairs = (len(topic_terms) * (len(topic_terms) - 1)) / 2
    topic_coh = topic_coh / num_pairs
    
    return topic_coh
    
def topic_coherence(bows_train: sparse.csr.csr_matrix, 
                    all_topic_ixs: List[List[int]], 
                    c_tend_func: Callable[[List], float] = np.median) -> Tuple[float, List[float]]:
    """
        Calculate the overall topic coherence given a source corpus and list of topics/terms
        
        Parameters:
            -bows_train
                Bag of words representations of all documents in the source corpus
            -all_topic_ixs
                The top n terms in each topic represented as word ids
            -c_tend_func: (List[float]) -> float
                The central tendency function to use across topics.
        Return: Tuple of (central_topic_coherence, all_topic_cohs)
            -central_topic_coherence: float
                Median or mean topic coherence depending on c_tend
            -all_topic_cohs: List[float]
                All individual topic coherence scores
        
    """
    
    cohs = []
    for topic_ixs in all_topic_ixs:
        top_coh = single_topic_coh(bows_train, topic_ixs)
        cohs.append(top_coh)
    
    return c_tend_func(cohs), cohs
    

# ----- TOPIC DIVERSITY -----

def topic_diversity(all_topic_ixs: List[List[int]]) -> float:
    """
        Calculate the topic diversity given a set of topics
        
        Parameters:
            -all_topic_ixs: List[List[int]]
                2-dimensional list of size (num_topics, num_top_terms) listing term ids
                    of most probable terms in each topic
                    
        Return: float
            The topic diversity score for the given set of topics
    """
        
    n_unique = len(np.unique(np.array(all_topic_ixs))) # Number of unique terms among topic list
    n_total  = len(all_topic_ixs)*len(all_topic_ixs[0]) # Total number of terms (num_topics * num_terms_per_topic)
    
    return n_unique/n_total



# ----- Combined Evaluation -----

def topic_eval(ref_bows: sparse.csr.csr_matrix,
               all_topic_ixs: List[List[int]], # Support tensors and np arrays?
               c_tend_func: Callable[[List], float] = np.median,
               return_all_cohs: bool = True
              ) -> Union[Tuple[float, float, float, List[float]], Tuple[float, float, float]]:
    """
        Calculate all topic evaluation metrics from a training corpus and topic terms
        
        Parameters:
            -ref_bows: sparse.csr_matrix
                Bag of words representations of all documents in the source/reference corpus
            -all_topic_ixs: List[List[int]]
                2-dimensional list of size (num_topics, num_top_terms) listing term ids
                    of most probable terms in each topic
            -c_tend_func: (List[float])->float, default = np.median
                Designates measure of central tendency to use in determining overall topic coherence
            -return_all_cohs: bool, default = True
                If true, returns all individual topic coherences in a list in the last index of
                    returned tuple
                    
        Return: Tuple of (topic_coherence, topic_diversity, topic_quality, [all_topic_cohs])
            -topic_coherence: float
                Overall topic coherence score
            -topic_diversity: float
                Overall topic diversity score
            -topic_quality: float
                Combination metric of topic coherence and diversity
                    (Calculated as coherence * diversity)
            -all_topic_cohs: List[float] 
                List of all individual topic coherences.
                    Only included if return_all_cohs is set to True
                
    """
    
    # Coherence
    coh, all_cohs = topic_coherence(ref_bows, all_topic_ixs, c_tend_func = c_tend_func)
    
    #Diversity
    div           = topic_diversity(all_topic_ixs)
    
    # Topic Quality
    quality       = coh * div
    
    if return_all_cohs:
        return div, coh, quality, all_cohs
    else:
        return div, coh, quality

def embed_coh(all_topic_ixs, embeds, reduce_func = np.median):
    
    cos_sims = []
    
    for topic in all_topic_ixs:
        topic_sims = []
        topic_embeds = embeds[topic]
        
        for i in range(topic_embeds.shape[0] - 1):
            for j in range(i + 1, topic_embeds.shape[0]):
                embed1, embed2 = topic_embeds[i], topic_embeds[j]
                cos_sim = F.cosine_similarity(embed1, embed2)
                
                topic_sims.append(cos_sim)
                
        topic_sim_c = reduce_func(topic_sims)
        cos_sims.append(topic_sim_c)
    
    return reduce_func(cos_sims), cos_sims