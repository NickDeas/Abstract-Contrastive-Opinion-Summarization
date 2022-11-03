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

from nltk.stem.porter import *
import spacy

# Initialize stemmer and spacy parser once
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

def read_and_process_data(source_fps: Union[str, list],
                          output_dir: str,
                          text_col: str = 'text',
                          test_frac: float = 0.,
                          val_frac: float = 0.,
                          vect_min_df: Union[int, float] = 10,
                          vect_max_df: Union[int, float] = 0.7,
                          min_toks: int = 2,
                          max_vocab: int = None,
                          embed_source: str = None,
                          should_split: bool = True,
                          ext_verbs: bool = True,
                          ev_batch_size = 500):
    """
        Process topic modeling dataset and save results in output directory
        
        Parameters:
            -source_fp: str
                File path to the source csv containing at least texts
            -output_dir: str
                Path to the output directory to store preprocessed data
            -text_col: str, default = 'text'
                Name of csv column containing full text
            -author_col: str, default = 'author'
                Name of csv column containing author name/identifier
            -test_frac: float, default = 0.2
                Fraction of data to randomly split as a test set
            -val_frac: float, default = 0.1
                Fraction of data to randomly split as a validation set
            -vect_min_df: int or float, default = 10
                min_df parameter to pass to CountVectorizer: the minimum number of documents a token
                    must appear in to be included in the vocabulary
            -vect_max_df: int or float, default = 0.7
                max_df parameter to pass to CountVectorizer: the maximum number of documents a token
                    can appear in to be included in the vocabulary
            -min_toks: int, default = 2
                Minimum number of tokens a document must contain to be preserved in the dataset
            -max_vocab: int, default = None
                The maximum number of tokens to keep in the vocabulary. If left as none, no maximum is set
        Return: None
    """
    
    if ext_verbs:
        o_vectorizers = {'verbs': None, 'nverbs': None}
        if embed_source:
            o_embeds = {'verbs': None, 'nverbs': None}
    
    print(f'Reading data from {source_fps}')
    
    dfs = {}
    if isinstance(source_fps, dict):
        for split, data_fp in source_fps.items():
            dfs[split] = pd.read_csv(data_fp)
            dfs[split]['len'] = dfs[split][text_col].str.len()
            
            if ext_verbs:
                print(f'Extracting verbs from {split}')
                dfs[split] = dfs[split].progress_apply(lambda row: extract_verbs_row(row, text_column = text_col), axis = 1)
                
    else:
        data = pd.read_csv(source_fps)
        data['len'] = data[text_col].str.len()
        if ext_verbs:
            print(f'Extracting verbs from data')
            data = data.progress_apply(lambda row: extract_verbs_row(row, text_column = text_col), axis = 1)
            
        if test_frac > 0:
            dfs['train'], dfs['test'] = train_test_split(data, test_size = test_frac + val_frac)   
            if val_frac > 0:
                dfs['test'], dfs['val']   = train_test_split(dfs['test'], test_size = val_frac/(test_frac + val_frac))
        else:
            dfs['train'] = data
    
    
    
    dfs = {
        split: df.sort_values('len', ascending = True)
        for split, df in dfs.items()
    }
    
    print(f'Cleaning text data {"and verb columns" if ext_verbs else ""}')
    for split, df in dfs.items():
        df['text_clean'] = df[text_col].astype(str).progress_apply(lambda text: clean.kitchen_sink(text))
        if ext_verbs:
            df['verbs_clean'] = df['verb_text'].astype(str).progress_apply(lambda text: clean.kitchen_sink(text))
            df['nverbs_clean'] = df['nverb_text'].astype(str).progress_apply(lambda text: clean.kitchen_sink(text))
    
    
    print(f'Fitting vectorizer')
    vectorizer = CountVectorizer(min_df = vect_min_df, max_df = vect_max_df, max_features = max_vocab)
    vectorizer = vectorizer.fit(dfs['train']['text_clean'].astype(str))
    print(f'Vectorizer trained with vocab size {len(vectorizer.vocabulary_)}')
    
    if ext_verbs:
        o_vectorizers['verbs'] = CountVectorizer()
        o_vectorizers['verbs'] = o_vectorizers['verbs'].fit(dfs['train']['verbs_clean'].astype(str))
        
        o_vectorizers['nverbs'] = CountVectorizer(min_df = vect_min_df, max_df = vect_max_df, max_features = max_vocab)
        o_vectorizers['nverbs'] = o_vectorizers['nverbs'].fit(dfs['train']['nverbs_clean'].astype(str))
                
    
    if embed_source:
        embeds = read_embeds(embed_source)
        vectorizer_filt, embed_mat_filt = align_vect_embed(vectorizer, embeds)
        
        if ext_verbs:
            o_vectorizers['verbs'], o_embeds['verbs'] = align_vect_embed(o_vectorizers['verbs'], embeds)
            o_vectorizers['nverbs'], o_embeds['nverbs'] = align_vect_embed(o_vectorizers['nverbs'], embeds)
            
        
        print(f'Filtered vectorizer and embeddings for {embed_mat_filt.shape[0]} tokens')
        
    
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
    
    
    
    if ext_verbs:
        for name, vec in o_vectorizers.items():
            with open(output_dir + f'/{name}_vectorizer.pkl', 'wb') as f:
                pkl.dump(vec, f)
    
    if embed_source:
        with open(output_dir + '/vectorizer_full.pkl', 'wb') as f:
            pkl.dump(vectorizer, f)

        with open(output_dir + '/vectorizer.pkl', 'wb') as f:
            pkl.dump(vectorizer_filt, f)

        np.save(output_dir + '/embeds.npy', embed_mat_filt)
        
        if ext_verbs:
            for name, o_embed in o_embeds.items():
                np.save(output_dir + f'/{name}_embeds.npy', o_embed)
    else:
        with open(output_dir + '/vectorizer.pkl', 'wb') as f:
            pkl.dump(vectorizer, f)
            
        if ext_verbs:
            for name, vec in o_vectorizers.items():
                with open(output_dir + f'/{name}_vectorizer.pkl', 'wb') as f:
                    pkl.dump(vec, f)
    
    
    params = {
        'test_frac': test_frac,
        'val_frac': val_frac,
        'vec_min_df': vect_min_df,
        'vec_max_df': vect_max_df,
        'min_toks': min_toks
    }
    
    with open(output_dir + '/params.json', 'w') as f:
        json.dump(params, f)
        
    print(f'Saved all data, vectorizer, and author mapping to {output_dir}')

def read_embeds(embed_source):
    print(f'Reading embeddings source')
    embeddings_dict = {}
    with open(embed_source, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    print(f'Read embeddings with {len(embeddings_dict)} terms')
    return embeddings_dict
    
def align_vect_embed(vectorizer, embeds):
    print(f'Filtering vectorizer and embeddings')
    inter_vocab = vectorizer.vocabulary_.keys() & embeds.keys()
    vectorizer_filt = CountVectorizer()
    vectorizer_filt.vocabulary_ = dict((t, i) for i, t in enumerate(vectorizer.vocabulary_.keys()) if t in inter_vocab)
    # Renumber terms
    vectorizer_filt.vocabulary_ = dict((t, i) for i, t in enumerate(vectorizer_filt.vocabulary_.keys()))

    # Extract embeddings in vectorizer vocabulary
    sub_embeds = dict((t, v) for t, v in embeds.items() if t in inter_vocab)
    embed_list = [sub_embeds[t] for t in vectorizer_filt.vocabulary_.keys()]
    embed_mat = np.stack(embed_list)
        
    print(f'Filtered vectorizer and embeddings for {embed_mat.shape[0]} tokens')
    
    return vectorizer_filt, embed_mat

def extract_verbs(text):
    
    parsed = nlp(text)
    
    verbs, nverbs = [], []
    
    for token in parsed:
        if token.pos_ == 'VERB':
            verbs.append(stemmer.stem(token.text))
        else:
            nverbs.append(token.text)
            
    return ' '.join(verbs), ' '.join(nverbs)

def extract_verbs_row(row, text_column = 'text'):
    verbs, nverbs = extract_verbs(str(row[text_column]))
    row['verb_text'] = verbs
    row['nverb_text'] = nverbs
    
    return row



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
    
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Old SRL Extractor codes  
#     def pred_to_terms_o(srl_preds, ev_join_char = ' '):
#     '''
#         Convert SRL parser predictions to event triples format
#     '''
    
#     verbs = srl_preds['verbs']
#     words = srl_preds['words']
#     triples = []
#     ignore_vs = []
#     ignore_indices = []
#     filt_sent, filt_sent_v = '', ''
    
#     for verb in verbs:
#         tags = verb['tags']
#         triple = ([], [], [], [])
#         ignore = []
#         ignore_v = []
        
#         for i, tag in enumerate(tags):
#             if tag[-2:] == '-V':
#                 triple[0].append(verb['lemma'])
#                 ignore.append(i)
#                 ignore_v.append(i)
#             elif tag[-5:] == '-ARG0':
#                 triple[1].append(stemmer.stem(words[i]))
#                 ignore.append(i)
#             elif tag[-5:] == '-ARG1':
#                 triple[2].append(stemmer.stem(words[i]))
#                 ignore.append(i)
#             elif tag[-5:] == '-ARG2':
#                 triple[3].append(stemmer.stem(words[i]))
#                 ignore.append(i)
                
#         if len(triple[1]) + len(triple[2]) + len(triple[3]) == 0:
#             continue
#         else:
#             ignore_indices += ignore
#             ignore_vs += ignore_v
            
#         triple = [ev_join_char.join(arg) for arg in triple]
#         triples.append(triple)
        
#     filt_sent   = ' '.join([word for i, word in enumerate(words) if not i in ignore_indices])
#     filt_sent_v = ' '.join([word for i, word in enumerate(words) if not i in ignore_vs])
        
#     return triples, filt_sent_v, filt_sent

# def pred_to_terms(srl_preds, ev_join_char = ' '):
    
#     events = srl_preds['verbs']
#     words  = srl_preds['words']
#     verb_text = ' '.join([event['lemma'] for event in events])
#     all_tags, all_words = [tag for event in events for tag in event['tags']], [word for event in events for word in words]
#     ev_text = ' '.join([stemmer.stem(word) for tag, word in zip(all_tags, all_words) if tag[-5:] in ('-ARG0', '-ARG1', '-ARG2')])
    
#     ev_text = verb_text + ' ' + ev_text
    
#     nv_text = ' '.join([stemmer.stem(word) for tag, word in zip(all_tags, all_words) if tag[-2:] != '-V'])
#     ne_text = ' '.join([stemmer.stem(word) for tag, word in zip(all_tags, all_words) if tag[-2:] not in ('G0', 'G1', 'G2', '-V')])
    
#     return verb_text, ev_text, nv_text, ne_text
    
    
    
# def text_to_triplets(row, preds = None):
#     '''
#         Extract triplets representing linguistic events from the given data row
#     '''
    
#     text = row['text']
#     try:
#         res = predictor.predict(sentence = text)
#         # triples, nv_text, ne_text = pred_to_terms(res)
#         v_text, e_text, nv_text, ne_text = pred_to_terms(res)
#     except Exception as e:
#         print(e)
#         # triples = []
#         v_text = ''
#         e_text = ''
#         nv_text = text
#         ne_text = text
    
#     # verbs = ' '.join([trip[0] for trip in triples])
#     # events = ' '.join(['_'.join(trip) for trip in triples])
    
#     row['verb_text'] = v_text #verbs
#     row['event_text'] = e_text #events
#     row['no_verb_text'] = nv_text
#     row['no_event_text'] = ne_text
    
#     return row

# def get_events(texts, batch_size = 200):
#     tot_time = 0
    
#     res = {key: [] for key in ('verb_text', 'event_text', 'no_verb_text', 'no_event_text')}
#     text_json = [{'sentence': text} for text in texts]
    
#     num_batches = len(text_json) // batch_size + 1
    
#     for batch_i in range(num_batches):
#         stime = time.time()
        
#         max_idx = max( ((batch_i + 1) * batch_size), len(text_json) )
#         subtext = text_json[batch_i * batch_size: max_idx]
#         try:
#             batch_res = predictor.predict_batch_json(subtext)
#             for i, text_res in enumerate(batch_res):
#                 # triples, nv_text, ne_text = pred_to_terms(text_res)
                
#                 # verbs = ' '.join([trip[0] for trip in triples])
#                 # events = ' '.join(['_'.join(trip) for trip in triples])
                
#                 v_text, e_text, nv_text, ne_text = pred_to_terms(text_res)

#                 res['verb_text'].append(v_text)
#                 res['event_text'].append(e_text)
#                 res['no_verb_text'].append(nv_text)
#                 res['no_event_text'].append(ne_text)
#         except Exception as e:
#             print(f'SRL Errored: {e}')
#             error_res = ['error' for j in range(len(batch_res))]
#             res['verb_text'] += error_res
#             res['event_text'] += error_res
#             res['no_verb_text'] += error_res
#             res['no_event_text'] += error_res
    
#         etime = time.time()
#         tot_time += (etime - stime)
#         print(f'\tCompleted batch {batch_i + 1} of {num_batches} in {etime - stime:0.3}s          ', end = '\r')
#     print(f'\tCompleted event text extraction                                         ')
        
#     return res

# def extract_events_df(df, batch_size = 200):
#     texts = df['text'].values
#     texts = [' '.join(text.split(' ')[:500]) for text in texts]
#     events = get_events(texts, batch_size = batch_size)
#     ev_df = pd.DataFrame(events)
    
#     new_df = pd.concat((df, ev_df), axis = 1)
#     return new_df

# def text_to_json(df):
#     texts = df['text'].values
#     json = [{'sentence': text} for text in texts]
    
#     return json