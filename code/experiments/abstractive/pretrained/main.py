# CLI script for running Abstractive Baselines (Currently only supports BART and PRIMERA based models)

import argparse
from tl_lib import *
from few_shot_run import *
from zero_shot_run import *


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description = 'Run Neural Baseline Tests for PoliSumm')

        sub_parsers = parser.add_subparsers(help = 'sub-command help')

        # Zero-Shot Evaluation Parameters
        parser_zs = sub_parsers.add_parser('zshot', help = 'Run a zero shot test without any training or evaluation')
        parser_zs.add_argument('-m', '--model', type = str, help = 'HuggingFace Model Checkpoint to use. (Must be Bart or Pegasus)')
        parser_zs.add_argument('-tfp', '--test-fp', type = str, help = 'Location of the test csv data')
        parser_zs.add_argument('-bs', '--batch-size', type = int, help = 'Batch size for training and testing', default = 4)
        parser_zs.add_argument('-rf', '--results-fp', type = str, help = 'Results directory')
        parser_zs.set_defaults(name = 'zshot')

        # Few-Shot/KFold Evaluation parameters
        parser_fs = sub_parsers.add_parser('fshot', help = 'Run a few shot test with limited training data')
        parser_fs.add_argument('-m', '--model', type = str, help = 'HuggingFace Model Checkpoint to use. (Must be Bart or Pegasus)')
        parser_fs.add_argument('-tr', '--train-fp', type = str, help = 'Location of the training csv data')
        parser_fs.add_argument('-te', '--test-fp', type = str, help = 'Location of the test csv data')
        parser_fs.add_argument('-bs', '--batch-size', type = int, help = 'Batch size for training and testing', default = 4)
        parser_fs.add_argument('-cd', '--chkpt-dir', type = str, help = 'Output Directory for Model Checkpoints')
        parser_fs.add_argument('-rf', '--results-fp', type = str, help = 'Results directory')
        parser_fs.set_defaults(name = 'fshot')

        args = vars(parser.parse_args())

        
        sc_name = args['name']
        chkpt_dir = args['chkpt_dir']
        res_dir   = args['results_fp']
        batch_size = args['batch_size']
        
        
        # Extract the model type from the checkpoint to pass to evaluation functions
        model_chkpt = args['model']
        if 'bart' in model_chkpt:
                model_type = 'bart'
        elif 'PRIMERA' in model_chkpt:
                model_type = 'primera'
        
        
        if sc_name == 'zshot':
                # Extract file path of test csv file
                test_fp = args['test_fp']
                
                # Call the zero shot evaluation function
                results = run_zero_shot(model_type, 
                                        model_chkpt, 
                                        test_fp, 
                                        chkpt_dir = chkpt_dir, 
                                        log_dir = res_dir, 
                                        batch_size = batch_size)


        elif sc_name == 'fshot':
                # Extract the data sources from arguments and create dictionary
                data_srcs = {'train': args['train_fp'],
                                        'test': args['test_fp']}
                
                # Call few shot evaluation function
                run_few_shot(model_type, 
                             model_chkpt, 
                             data_srcs, 
                             chkpt_dir = chkpt_dir, 
                             log_dir = res_dir, 
                             batch_size = batch_size) 






