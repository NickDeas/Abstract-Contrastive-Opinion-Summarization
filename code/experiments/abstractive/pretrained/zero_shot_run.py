from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, LEDForConditionalGeneration
import pytorch_lightning as pl
from tl_lib import *

def run_zero_shot(model_type, 
                  model_chkpt, 
                  test_fp, 
                  chkpt_dir = './checkpoints', 
                  log_dir = './', 
                  batch_size = 4,
                  num_devices = 1,
                  max_epochs = 3,
                  min_epochs = 1,
                  learning_rate = 3e-5):
        '''
            Run a zero-shot evaluation
            
            Parameters:
                -model_type:str
                    The type of pre-trained model used for evaluation ('bart' or 'primera')
                -model_chkpt: str
                    The model checkpoint on HuggingFace hub to use
                -test_fp: str
                    File path to the testing dataset
                -chkpt_dir: str
                    Directory to store trained model checkpoints
                -log_dir: str
                    Directory to store lightning log results
                -batch_size: int
                    Batch size for model training
                -num_devices: int
                    Number of GPUs to use during evaluation
                -max_epochs: int
                    Maximum number of epochs to train for
                -min_epochs: int
                    Minimum number of epochs to train for
                -learning_rate: float
                    Learning rate to use in training
            
            Return
                None
        '''
    
        print('Loading Model')
        
        # Load Model and Tokenizer depending on model type/checkpoint
        if model_type == 'bart':
                model = BartForConditionalGeneration.from_pretrained(model_chkpt)
                tokenizer = BartTokenizer.from_pretrained(model_chkpt)
        elif model_type == 'primera':
                model = LEDForConditionalGeneration.from_pretrained(model_chkpt)
                tokenizer = AutoTokenizer.from_pretrained(model_chkpt)

        print('Setting Up Lightning Model')
        lightning_model = PoliSummModel(tokenizer, model, learning_rate = learning_rate)

        print('Setting Up Data Module')
        data_mod = PoliSummEvalModule(tokenizer, test_fp, batch_size = batch_size, is_primera = (model_type == 'primera'))
        data_mod.prepare_data()
        data_mod.setup()

        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=chkpt_dir)

        print('Creating Trainer')
        # Create a logger to output loss and metrics during training
        tb_logger = pl.loggers.TensorBoardLogger(save_dir = log_dir)
        
        trainer = pl.Trainer(gpus = num_devices,
                     max_epochs = max_epochs,
                     min_epochs = min_epochs,
                     auto_lr_find = False,
                     callbacks = [checkpoint],
                     val_check_interval = 0.25,
                     log_every_n_steps = 5,
                     logger = tb_logger)

        print('Beginning Testing')
        # run testing and generate example summaries
        trainer.test(lightning_model, datamodule = data_mod)

        print('Done Training and Testing')

