from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import LEDForConditionalGeneration, AutoTokenizer
import pytorch_lightning as pl
from tl_lib import *

def run_few_shot(model_type, 
                 model_chkpt, 
                 data_srcs, 
                 chkpt_dir = './checkpoints', 
                 log_dir = './results', 
                 batch_size = 4, 
                 num_devices = 1, 
                 max_epochs = 3, 
                 min_epochs = 1):
            '''
            Run training training and evaluation
            
            Parameters:
                -model_type:str
                    The type of pre-trained model used for evaluation ('bart' or 'primera')
                -model_chkpt: str
                    The model checkpoint on HuggingFace hub to use
                -data_srcs: dict
                    Dictionary with file paths to training, validation, and test datasets
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
        # Load Model and Tokenizer
        if model_type == 'bart':
                model = BartForConditionalGeneration.from_pretrained(model_chkpt)
                tokenizer = BartTokenizer.from_pretrained(model_chkpt)
        elif model_type == 'primera':
                model = LEDForConditionalGeneration.from_pretrained(model_chkpt)
                tokenizer = AutoTokenizer.from_pretrained(model_chkpt)

        print('Setting Up Lightning Model')
        lightning_model = PoliSummModel(tokenizer, model, learning_rate = learning_rate)

        print('Setting Up Data Module')
        data_mod = PoliSummDataModule(tokenizer, data_srcs, batch_size = batch_size)
        data_mod.prepare_data()
        data_mod.setup()

        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=chkpt_dir)

        print('Creating Trainer')
        tb_logger = pl.loggers.TensorBoardLogger(save_dir = log_dir)
        trainer = pl.Trainer(accelerator = 'gpu',
                     devices = num_devices,
                     max_epochs = max_epochs,
                     min_epochs = min_epochs,
                     auto_lr_find = False,
                     callbacks = [checkpoint],
                     val_check_interval = 0.25,
                     log_every_n_steps = 1,
                     logger = tb_logger)

        print('Beginning Training')
        train_results = trainer.fit(lightning_model, datamodule = data_mod)

        print('Beginning Testing')
        test_results = trainer.test(lightning_model, datamodule = data_mod)

        print('Done Training and Testing')

        return test_results
