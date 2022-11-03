import argparse
from tl_lib import *
from few_shot_run import *
from zero_shot_run import *


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description = 'Run Neural Baseline Tests for PoliSumm')

        sub_parsers = parser.add_subparsers(help = 'sub-command help')

        parser_zs = sub_parsers.add_parser('zshot', help = 'Run a zero shot test without any training or evaluation')
        parser_zs.add_argument('-m', '--model', type = str, help = 'HuggingFace Model Checkpoint to use. (Must be Bart or Pegasus)')
        parser_zs.add_argument('-tfp', '--test-fp', type = str, help = 'Location of the test csv data')
        parser_zs.add_argument('-bs', '--batch-size', type = int, help = 'Batch size for training and testing', default = 4)
        parser_zs.add_argument('-cd', '--chkpt-dir', type = str, help = 'Output Directory for Model Checkpoints')
        parser_zs.add_argument('-rf', '--results-fp', type = str, help = 'Results directory')
        parser_zs.set_defaults(name = 'zshot')

        parser_fs = sub_parsers.add_parser('fshot', help = 'Run a few shot test with limited training data')
        parser_fs.add_argument('-m', '--model', type = str, help = 'HuggingFace Model Checkpoint to use. (Must be Bart or Pegasus)')
        parser_fs.add_argument('-tr', '--train-fp', type = str, help = 'Location of the training csv data')
        parser_fs.add_argument('-te', '--test-fp', type = str, help = 'Location of the test csv data')
        parser_fs.add_argument('-v', '--val-fp', type = str, help = 'Location of the validation csv data')
        parser_fs.add_argument('-bs', '--batch-size', type = int, help = 'Batch size for training and testing', default = 4)
        parser_fs.add_argument('-cd', '--chkpt-dir', type = str, help = 'Output Directory for Model Checkpoints')
        parser_fs.add_argument('-rf', '--results-fp', type = str, help = 'Results directory')
        parser_fs.set_defaults(name = 'fshot')

        args = vars(parser.parse_args())

        sc_name = args['name']

        model_chkpt = args['model']
        if 'bart' in model_chkpt:
                model_type = 'bart'
        elif 'PRIMERA' in model_chkpt or 'cocosum' in model_chkpt:
                model_type = 'primera'
        elif 'pegasus' in model_chkpt:
                model_type = 'pegasus'

        chkpt_dir = args['chkpt_dir']
        res_dir   = args['results_fp']
        batch_size = args['batch_size']

        if sc_name == 'zshot':

                test_fp = args['test_fp']

                results = run_zero_shot(model_type, model_chkpt, test_fp, chkpt_dir = chkpt_dir, log_dir = res_dir, batch_size = batch_size)


        elif sc_name == 'fshot':

                data_srcs = {'train': args['train_fp'],
                                        'test': args['test_fp'],
                                        'val': args['val_fp']}

                run_few_shot(model_type, model_chkpt, data_srcs, chkpt_dir = chkpt_dir, log_dir = res_dir, batch_size = batch_size) 





