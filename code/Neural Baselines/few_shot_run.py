from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pytorch_lightning as pl
from tl_lib import *

def run_few_shot(model_type, model_chkpt, data_srcs, chkpt_dir = './checkpoints', batch_size = 4):

	print('Loading Model')
	# Load Model and Tokenizer
	if model_type == 'bart':
		model = BartForConditionalGeneration.from_pretrained(model_chkpt)
		tokenizer = BartTokenizer.from_pretrained(model_chkpt)
	elif model_type == 'pegasus':
		model = PegasusForConditionalGeneration.from_pretrained(model_chkpt)
		tokenizer = PegasusTokenizer.from_pretrained(model_chkpt)

	print('Setting Up Lightning Model')
	lightning_model = PoliSummModel(tokenizer, model)

	print('Setting Up Data Module')
	data_mod = PoliSummDataModule(tokenizer, data_srcs, batch_size = batch_size)
	data_mod.prepare_data()
	data_mod.setup()

	checkpoint = pl.callbacks.ModelCheckpoint(dirpath=chkpt_dir)

	print('Creating Trainer')
	trainer = pl.Trainer(gpus = 1,
                     max_epochs = 3,
                     min_epochs = 1,
                     auto_lr_find = False,
                     checkpoint_callback = checkpoint,
                     progress_bar_refresh_rate = 500,
                     val_check_interval = 0.25,
                     log_every_n_steps = 1)

	print('Beginning Training')
	trainer.fit(lightning_model, datamodule = data_mod)

	print('Beginning Testing')
	trainer.test(lightning_model, datamodule = data_mod)

	print('Done Training and Testing')