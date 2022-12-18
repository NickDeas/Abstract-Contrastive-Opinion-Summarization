import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.generation_utils import top_k_top_p_filtering, BeamSearchScorer
from transformers.pytorch_utils import torch_int_div

class ToGLDecoder:
    '''
        Wrapper class for a pre-trained language model implementing ToGL-Decoding
    '''
    
    def __init__(self,
                model: PreTrainedModel,
                tokenizer,
                top_p: float = 1.0,
                togl_func: str = 'sum',
                togl_func_kwargs: dict = None,
                device = None):
        '''
            Parameters:
                -model: PreTrainedModel
                    A Huggingface pretrained model capable of generating text
                -top_p: float
                    Parameter for top-p sampling decoding method
                -togl_func
                    Function used to combine model predictions and topic model word distribution.
                    Defaults to the sum of the generation and topic model word distributions with weight 1.
                -togl_func_kwargs
                    Keyword arguments to pass to the togl_func beyond word distribution parameters
                -device
                    Torch/Cuda device to use while generating
        '''
        
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = self.model.lm_head.out_features
        
        self.top_p = top_p
        
        # If a string, set the togl_func to one that is already implemented
        #      Leaves room for later addition of other compoisition functions
        # If not a string (custom function) 
        if type(togl_func) == str:
            assert togl_func in ('sum'), f'togl_func {togl_func} has not been implemented'
            if togl_func == 'sum':
                self.togl_func = self.togl_sum
        else:
            self.togl_func = togl_func
        
        self.togl_func_kwargs = togl_func_kwargs
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # START: Code partially adapted from
    #     https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/generation/utils.py#L998
    #     https://github.com/megagonlabs/cocosum/blob/main/decode.py
    # 
    @torch.no_grad()
    def generate(self,
                    inputs: torch.Tensor,
                    togl_tuple: tuple,
                    togl_start: int = 2,
                    togl_weight: int = 0.1,
                    decoder_start_token_id = None,
                    num_beams: int  = 3,
                    no_repeat_ngram_size = 3,
                    min_length: int = 16,
                    max_length: int = 1024,
                    early_stopping: bool = True,
                    **model_kwargs):
        '''
            Generates a sequence using beam search sampling incorporating ToGL-Decoding.
            
            Parameters:
                -inputs
                    Pytorch tensor of encoded and tokenized input sequences
                -togl_tuple
                    Tuple containing the ids and probabilities of terms to be made more probable
                -togl_start
                    What token index in the generation to being influencing word distributions
                    Help to prevent incoherent generations with non-sensical starting tokens
                -togl_weight
                    Weighting of ToGL distributions before using in composition function
                    Weights between 0.1 and 0.5 have been found to be an appropriate range
                -decoder_start_token_id
                    Custom token to start decoder sequences as in beam search generation in PyTorch
                -num_beams
                    Number of beams to use in underlying beam search decoding
                -no_repeat_ngram_size
                    Size of ngram that should not be repeated in generation beams
                -min_length
                    Minimum length of the generated text
                -max_length
                    Maximum length of the generated text
                -early_stopping
                    Whether generation should continue to max length or stop before according to other parameters
                -**model_kwargs
                    Any keyword arguments to pass to the language model and setup functions
        '''
        
        batch_size = 1
        
        inputs           = inputs.to(self.device)
        
        # Convert ToGL ids and probabilities to tensors in the shape of the language model vocab
        togl_probs       = self.togl_convert(togl_tuple, togl_weight)
        togl_probs       = togl_probs.to(self.device)
        
        # Preprocesses model inputs and model parameters for inference
        inputs_t, model_input_name, model_kwargs = self.model._prepare_model_inputs(inputs, self.tokenizer.bos_token_id, model_kwargs)
        model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_t, model_kwargs, model_input_name
        )
        
        # Preprocesses decoder inputs for text generation
        input_ids = self.model._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                model_kwargs=model_kwargs,
                device=self.device,
            )
        
        # Module to filter invalid (nan, inf) probabilities to avoid errors during generations
        logits_processor = self.model.model._get_logits_processor(
            repetition_penalty = None,
            no_repeat_ngram_size = no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=None,
            input_ids_seq_length = input_ids.shape[-1],
            encoder_input_ids = inputs_t,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            num_beams=num_beams,
            num_beam_groups=None,
            diversity_penalty=None,
            remove_invalid_values=None,
            bad_words_ids = None,
            prefix_allowed_tokens_fn = None,
            exponential_decay_length_penalty = None,
            logits_processor = [],
            renormalize_logits = None,
        )
        
        # Initialize the stopping criteria for generation
        stopping_criteria = self.model.model._get_stopping_criteria(
            max_length = max_length, max_time = None, stopping_criteria = []
        )
        
        # Setup beam scorer for searching generation candidates
        beam_scorer = BeamSearchScorer(
            batch_size = batch_size,
            num_beams = num_beams,
            device = self.device,
            do_early_stopping = early_stopping,
            num_beam_hyps_to_keep = 1
        )
        
        # Reformat inputs for beam search decoding with specified number of beams
        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=True, **model_kwargs
        )
        
        # Initialize beam search parameters, scores, and necessary variables
        batch_size = len(beam_scorer._beam_hyps)
        batch_beam_size, cur_len = input_ids.shape
        beam_scores = torch.zeros((batch_size, num_beams), 
                                  dtype = torch.float, 
                                  device = self.device)
        beam_scores[:, 1:] = -1e-9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        beam_indices = (None)
        
        # Generate tokens until stopping criteria is met
        while True:            
            
            # Preprocess inputs specifically for generation (torch util)
            model_in = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # Predict the next token in the generation
            outputs = self.model(**model_in, 
                                 return_dict=True,
                                 output_attentions = False,
                                 output_hidden_states = False)
            
            # Update togl_logits to zero out already generated terms
            togl_probs[input_ids] = 0.
                        
            # Modify logits with the ToGL Composition Function
            raw_logits = outputs.logits
            if cur_len >= togl_start:
                mod_logits = self.togl_func(raw_logits, togl_probs)
            else:
                mod_logits = raw_logits
            
            mod_logits = mod_logits[:, -1, :]
            
            # Adjust and transform logits into log probabilities (torch util)
            next_logits = self.model.model.adjust_logits_during_generation(mod_logits, cur_len = cur_len)
            next_scores = F.log_softmax(next_logits, dim = -1)
            
            # Modify invalid log probabilities to prevent errors
            next_scores_pp = logits_processor(input_ids, next_scores)
            next_scores = next_scores_pp + beam_scores[:, None].expand_as(next_scores)
            
            vocab_size  = next_scores.shape[-1]
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # Fetch the top k elements of the next predicted token distribution
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim = 1, largest = True, sorted = True
            )
            next_idxs = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size
            
            # Score the beam search candidates
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_idxs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                beam_indices=beam_indices,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            
            # Update inputs and model for the next token to generate (torch util)
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim = -1)
            model_kwargs = self.model.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder = True
            )
            
            cur_len += 1
            
            # Stop generations when stopping criteria is met
            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break
        
        # Finalize sequence among beam candidates and scoring
        seq_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_idxs,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            max_length = stopping_criteria.max_length,
            beam_indices = beam_indices,
        )
        
        # Return the final selected generated sequence
        return seq_outputs['sequences']
    
    def togl_sum(self, raw_out, togl_probs):
        '''
            Initial composition function for ToGL-Decoding.
            Conserves the norm of the raw outputs but performs composition in unit length space.
            ToGL Probabilities should already be scaled by togl_convert
            
            Parameters:
                -raw_out
                    Raw output tensor word distribution from the langauge model at a single time step
                -togl_probs
                    ToGL word distribution from the trained TAM to compose with the language model outputs
        '''
        
        # Calculate raw output logit norms
        norms = raw_out.norm(dim = -1)
        
        # Reshape and format togl_probs for the composition
        togl_probs = togl_probs.unsqueeze(0).repeat((raw_out.shape[0], 1))
        
        # Calculate modified logits adding the togl probabilities to the normed raw logits, rescaling by norm at the end
        mod_logits = ((raw_out.squeeze()/norms) + togl_probs) * norms
        mod_logits = mod_logits.unsqueeze(1)
        
        return mod_logits
    
    def togl_convert(self, togl_tuple, togl_weight):
        '''
            Prepare togl distributions for composition and generation
            
            Parameters:
                -togl_tuple
                    Tuple of term indices and weights output by the Topic-Aspect Model
                -togl_weight
                    Scale parameter to control influence of togl_decoding
            
            Return
                Expanded togl distribution to the shape of the language model vocab and scaled by togl_weight
        '''
        
        # Initialize vocab-sized tensor
        full_dist = torch.zeros((self.vocab_size), device = self.device)
        
        # Extract probabilities and indices
        probs = togl_tuple[0]
        idxs  = togl_tuple[1]
        
        # Set indices of zeros tensor to togl_probs
        full_dist[idxs] = probs
        
        # If the togl distribution is not all zeros, then scale by togl_weight
        if full_dist.sum() > 0:
            full_dist = (full_dist / full_dist.sum()) * togl_weight
        
        return full_dist
