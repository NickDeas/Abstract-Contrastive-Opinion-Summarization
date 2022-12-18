# CLI Script for running ToGL-Decoding Inference

import argparse
import json

import pandas as pd
from tqdm import tqdm

from togl_decoder import ToGLDecoder

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

MAX_INPUT_LEN = 1022

if __name__ == '__main__':
    
    print('Parsing arguments')
    parser = argparse.ArgumentParser(description = 'Run Togl-Decoding for PoliSum')
    
    parser.add_argument('-i', '--input',
                        type = str,
                        help = 'File path of input csv containing input texts',
                        required = True)
    parser.add_argument('-sc', '--src-col',
                        type = str,
                        help = 'Column name in input containing source texts',
                        required = True)
    parser.add_argument('-tl', '--togl-left',
                        type = str,
                        help = 'File path to json dictionary of togl distributions for left summaries',
                        required = True)
    parser.add_argument('-tr', '--togl-right',
                        type = str,
                        help = 'File path to json dictionary of togl distributions for right summaries',
                        required = True)
    parser.add_argument('-o', '--output',
                        type = str,
                        help = 'Output csv path to store generated summaries',
                        required = True)
    parser.add_argument('-m', '--model',
                        type = str,
                        help = 'Base pretrained model checkpoint to use in togl decoding',
                        required = False,
                        default = 'facebook/bart-large-xsum')
    parser.add_argument('-tf', '--togl-func',
                        type = str,
                        help = 'Function to use in combining output word distributions and ToGL distributions',
                        required = False,
                        default = 'sum')
    parser.add_argument('-ts', '--togl-start',
                        type = int,
                        help = 'Minimum index of token when togl distributions are incorporated',
                        required = False,
                        default = 3)
    parser.add_argument('-tw', '--togl_weight',
                        type = float,
                        help = 'Weighting of togl distributions in generation (should be less than 1)',
                        required = False,
                        default = 0.1)
    parser.add_argument('-nb', '--num-beams',
                        type = int,
                        help = 'Number of beams to use in beam search during generation',
                        default = 3)
    parser.add_argument('-nr', '--no-repeat-ngram',
                        type = int,
                        help = 'No Repeat Ngram Size to constrain generations',
                        required = False,
                        default = 3)
    parser.add_argument('-mi', '--min-length',
                        type = int,
                        help = 'Minimum length of the generation',
                        required = False,
                        default = 16)
    parser.add_argument('-ma', '--max-length',
                        type = int,
                        help = 'Maximum length of the generation',
                        required = False,
                        default = 128)
    parser.add_argument('-d', '--device',
                    type = str,
                    help = 'torch cuda device to use in generation',
                    required = False,
                    default = 'cuda:0')
    
    # Parse Command Line Arguments
    print('1. Parsing Arguments')
    args = vars(parser.parse_args())
    
    device = torch.device(args['device'])
    
    # Download model and tokenizer from HuggingFace Hub or cache if already downloaded
    print('2. Downloading model and tokenizer')
    # Load model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(args['model'])
    model     = BartForConditionalGeneration.from_pretrained(args['model']).to(device)
    
    
    # Instantiate ToGL Decoder
    togl_decoder = ToGLDecoder(model, tokenizer, device = torch.device(device))
    
    print('3. Reading data and ToGL distributions')
    # Read data
    data = pd.read_csv(args['input'])
    
    # Read stored togl distributions
    with open(args['togl_left'], 'r') as f:
        togl_l_dists = json.load(f)
    with open(args['togl_right'], 'r') as f:
        togl_r_dists = json.load(f)
    
    print('4. Generating summaries')
    # Iterate over rows and generate summaries
    outputs = {'title_date': [], 'l': [], 'r': []}
    for row in tqdm(data.to_dict(orient = 'records')):
        src_text = row[args['src_col']]
        src_title_date = row['title'] + '_' + row['date']
        
        # Create dictionary of left and right togl_distributions
        togl_dist = {
            'l': (torch.tensor(togl_l_dists[src_title_date][0]).to(device), torch.tensor(togl_l_dists[src_title_date][1], dtype = torch.long).to(device)),
            'r': (torch.tensor(togl_r_dists[src_title_date][0]).to(device), torch.tensor(togl_r_dists[src_title_date][1], dtype = torch.long).to(device)),
        }
        
        # Tokenize and encode inputs
        src_tok = tokenizer(src_text, padding = True, truncation = True, return_tensors = 'pt')
        src_tok = src_tok.to(device)
        
        # Generate a left and right summary using ToGL-Decoding
        for i, side in enumerate('lr'):
            output = togl_decoder.generate(src_tok['input_ids'],
                                           togl_dist[side],
                                           togl_start = args['togl_start'],
                                           togl_weight = args['togl_weight'],
                                           min_length = args['min_length'],
                                           max_length = args['max_length'],
                                           no_repeat_ngram_size = args['no_repeat_ngram'],
                                           num_beams = args['num_beams']
                                          )
            summary = tokenizer.decode(output[0], skip_special_tokens = True)
            outputs[side].append(summary)
        
        # Append the title_date to map back to PoliSum dataset
        outputs['title_date'].append(src_title_date)
    
    # Store generated summaries as csv
    print(f'Decoding complete. Saving generated summaries to {args["output"]}')
    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv(args['output'], index = None)
    
