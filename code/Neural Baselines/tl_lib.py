import torch
from torch.utils.data import  DataLoader

import pytorch_lightning as pl

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bert import BERTScore

import pandas as pd
import numpy as np


def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=1024, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
      Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
      Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        encoded_dict = tokenizer(
              sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
          )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
              sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
          )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)


    batch = {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
      "labels": target_ids,
    }

    return batch

# ----- Dataset -----
class PoliSummDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings):
        
        self.encodings = encodings
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])


# ----- Lightning Data Modules -----
class PoliSummEvalModule(pl.LightningDataModule):
    
    def __init__(self, tokenizer, src_csv, batch_size = 4):
        '''
            Data expected to have columns ('all_texts', 'all_sum')
        '''
        
        super().__init__()
        
        self.tokenizer = tokenizer
        self.src_csv = src_csv
        self.batch_size = batch_size
        
    def prepare_data(self):
        
        self.data = pd.read_csv(self.src_csv)
        assert 'all_texts' in self.data.columns, 'Missing source texts column: all_texts'
        assert 'all_sum' in self.data.columns, 'Missing target summaries column: all_sum'
        
    def setup(self, stage = None):

        src_texts = self.data['all_texts'].astype(str).values
        targ_texts = self.data['all_sum'].astype(str).values
        
        self.test_encodings = encode_sentences(self.tokenizer, src_texts, targ_texts, return_tensors = 'pt')
        
    def test_dataloader(self):
        dataset = PoliSummDataset(self.test_encodings)
        test_dl = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = False,
                              num_workers = 2)
        return test_dl

class PoliSummDataModule(pl.LightningDataModule):
    
    def __init__(self, tokenizer, src_csvs, batch_size = 4):
        '''
            Data expected to have columns ('all_texts', 'all_sum')
        '''
        
        super().__init__()
        
        self.tokenizer = tokenizer
        self.src_csvs = src_csvs
        self.batch_size = batch_size
        
    def prepare_data(self):
        
        self.data_dict = {key: pd.read_csv(val) for key, val in self.src_csvs.items()}
        
    def setup(self, stage = None):
        
        train_src, train_targ = self.df_to_pairs(self.data_dict['train'])
        test_src, test_targ   = self.df_to_pairs(self.data_dict['test'])
        val_src, val_targ     = self.df_to_pairs(self.data_dict['val'])
        
        self.train_encodings = encode_sentences(self.tokenizer, train_src, train_targ, return_tensors = 'pt')
        self.test_encodings  = encode_sentences(self.tokenizer, test_src, test_targ, return_tensors = 'pt')
        self.val_encodings   = encode_sentences(self.tokenizer, val_src, val_targ, return_tensors = 'pt')
    
    def train_dataloader(self):
        dataset = PoliSummDataset(self.train_encodings)
        train_dl = DataLoader(dataset, 
                              batch_size = self.batch_size, 
                              shuffle = True,
                              num_workers = 2)
        return train_dl
        
    def test_dataloader(self):
        dataset = PoliSummDataset(self.test_encodings)
        test_dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = False,
                              num_workers = 2)
        return test_dl
    
    def val_dataloader(self):
        dataset = PoliSummDataset(self.val_encodings)
        val_dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = False,
                              num_workers = 2)
    
    def df_to_pairs(self, df):
        src_l = 'summarize left: ' + df['all_texts'].astype(str)
        src_r = 'summarize right: ' + df['all_texts'].astype(str)
        src_texts_double = np.concatenate((src_l.values, src_r.values))
        
        targ_l    = df['left_sum'].astype(str).values
        targ_r    = df['right_sum'].astype(str).values
        targs     = np.concatenate((targ_l, targ_r))
        
        return src_texts_double, targs
        


# ----- Lightning Model -----
class PoliSummModel(pl.LightningModule):
    
    def __init__(self, tokenizer, model, test_in_train = True):
        
        super().__init__()
        
        self.tokenizer = tokenizer
        self.model = model
        self.test_in_train = test_in_train
        
        if self.test_in_train:
            self.rouge_scorer = ROUGEScore()
            self.bert_scorer  = BERTScore()

        
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        src, mask, targ = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        output = self(src, 
                     attention_mask = mask,
                     decoder_input_ids = targ)
        logits = output[0]
        
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index = self.tokenizer.pad_token_id)
        train_loss = ce_loss(logits, targ)
        
        return {'loss': train_loss}
    
    def validation_step(self, batch, batch_idx):
        src, mask, targ = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        output = self(src, 
                     attention_mask = mask,
                     decoder_input_ids = targ)
        logits = output[0]
        
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index = self.tokenizer.pad_token_id)
        val_loss = ce_loss(logits, targ)
        
        res_dict = {'loss': val_loss}
        
        # Summarization Metrics
        if self.test_in_train:
            gen_summs = self.generate_summ(batch, max_len = 256)
            ref_summs = self.untokenize_targ(batch)
            
            # ROUGE Score
            rouge_keep = ('rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure')
            rouge_scores = self.rouge_scorer(gen_summs, ref_summs)
            rouge_scores = {name:score for name, score in rouge_scores if name in rouge_keep}          
            
            res_dict.update(rouge_scores)
        
        return res_dict
    
    def test_step(self, batch, batch_idx):
        src, mask, targ = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        output = self(src, 
                     attention_mask = mask,
                     decoder_input_ids = targ)
        logits = output[0]
        
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index = self.tokenizer.pad_token_id)
        test_loss = ce_loss(logits, targ)
        
        res_dict = {'loss': test_loss}
        gen_summs = self.generate_summ(batch, max_len = 256)
        ref_summs = self.untokenize_targ(batch)
        
        # ROUGE Score
        rouge_keep = ('rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure')
        rouge_scores = self.rouge_scorer(gen_summs, ref_summs)
        rouge_scores = {name:score for name, score in rouge_scores if name in rouge_keep}          

        res_dict.update(rouge_scores)
        
        # BERT Score
        bert_scores = self.bert_scorer(gen_summs, ref_summs)
        bert_scores = {'bert_' + key:val for key, val in bert_scores.items()}
        res_dict.update(bert_scores)
        
        return res_dict
    
    def generate_summ(self, sample, eval_beams = 3, early_stopping = True, max_len = 40):

        generated_ids = self.model.generate(
            sample["input_ids"],
            attention_mask=sample["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= eval_beams,
            max_length = max_len,
            early_stopping = early_stopping
        )
        
        gen_summ = [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]
        return gen_summ
    
    def untokenize_targ(self, sample):
        sent      = sample['labels']
        targ_summ = [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in sent]
        return targ_summ
        
        
        