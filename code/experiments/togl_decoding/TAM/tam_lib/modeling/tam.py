import os
import pickle as pkl
import json

import torch
from torch import nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .layers import *

from typing import Optional, List, Union, Tuple


class TAM(nn.Module):
    def __init__(self, num_topics:int, num_aspects: int,
                 hidden_size: int = None,
                 vocab_size: int = None,
                 dropout: float = None,
                 delta_encoder: Encoder = None, tau_encoder: Encoder = None, 
                 top_decoder: Decoder = None, asp_decoder: Decoder = None, ta_decoder: JointDecoder = None,
                 bg_topic: torch.Tensor = None,
                 delta_prior: pyro.distributions.torch_distribution.TorchDistributionMixin = None,
                 tau_prior: pyro.distributions.torch_distribution.TorchDistributionMixin = None,
                 sigma_prior: pyro.distributions.torch_distribution.TorchDistributionMixin = None,
                 delta_scale: float = 3., tau_scale:float = 3.,
                 gamma_0: float = 10., gamma_1: float = 10.,
                 sigma_0: float = 10., sigma_1: float = 10.,
                 device: torch.device = None):
        """
            Topic Aspect Model implementation including model and guide.
            
            Parameters:
                num_topics: int
                    Number of latent topics assumed
                num_aspects: int
                    Number of latent aspects assumed
                hidden_size: int (optional)
                    Size of the hidden layer in the Encoder
                vocab_size: int (optional)
                    Number of tokens in the vocabulary
                dropout: float (optional)
                    Dropout rate for the encoder
                delta_encoder: models.layers.Encoder (optional)
                    Encoder layer(s) transforming document BOW's into an unnormalized topic distribution
                tau_encoder: models.layers.Encoder (optional)
                    Encoder layer(s) transforming document BOW's into a unnormalized aspect distribution
                top_decoder: models.layers.Decoder (optional)
                    Decoder layer(s) transforming topic distribution into a document BOW.
                    Corresponds to the Beta matrix in ProdLDA num_topics x vocab_size
                asp_decoder: models.layers.Decoder (optional)
                    Decoder layer(s) transforming topic distribution into a document BOW.
                    Corresponds to a Beta matrix applied to aspects num_aspects x vocab_size
                ta_decoder: models.layers.Decoder (optional)
                    Decoder layer(s) transforming joint topic/aspect distribution into a document BOW.
                    Corresponds to a Beta matrix applied to both topics and aspects (num_topics*num_aspects) x vocab_size
                delta_prior: pyro.distributions (optional)
                    Prior distribution on the latent topics
                tau_prior: pyro.distributions (optional)
                    Prior distribution on the latent aspects
                sigma_prior: pyro.distributions (optional)
                    Prior distribution on the binomial level distribution
                delta_scale: float
                    If no delta_prior passed, the scale/variance of the LogNormal prior placed on delta (unnormalized topic distribution)
                tau_scale: float
                    If no tau_prior passed, the scale/variance of the LogNormal prior placed on tau (unnormalized aspect distribution)
                gamma_0, gamma_1: float
                    Parameters of the binomial distribution over paths in the TAM
                sigma_0, sigma_1: float
                    Parameters of the binomial distribution over levels in the TAM
                device: torch.device (optional)
                    GPU/CPU device to use in training/inference
        """
        
        super().__init__()   
        
        # Latent size
        self.num_topics = num_topics
        self.num_aspects = num_aspects
        
        if not ((hidden_size and vocab_size and dropout) or (delta_encoder and tau_encoder)):
            raise Exception('Either hidden_size, vocab_size, and dropout or an encoder must be specified.')
        if not ((vocab_size and dropout) or (top_decoder and asp_decoder and ta_decoder)):
            raise Exception('Either vocab_size or a decoder must be specified')
        
        # Encoder/Decoder parameters
        self.hidden_size = hidden_size if hidden_size else delta_encoder.hidden_size
        self.vocab_size  = vocab_size if vocab_size else delta_encoder.vocab_size
        self.dropout     = dropout if dropout else delta_encoder.dropout
                
        # Encoders
        self.delta_encoder  = delta_encoder if delta_encoder else Encoder(vocab_size, num_topics, hidden_size, dropout)
        self.tau_encoder    = tau_encoder if tau_encoder else Encoder(vocab_size, num_aspects, hidden_size, dropout)
        
        # Decoders
        self.top_decoder = top_decoder if top_decoder else Decoder(vocab_size, num_topics, dropout)
        self.asp_decoder = asp_decoder if asp_decoder else Decoder(vocab_size, num_aspects, dropout)
        self.ta_decoder  = ta_decoder  if ta_decoder  else JointDecoder(vocab_size, num_topics, num_aspects, dropout)
        
        # Device
        self.device = device if device else torch.device("cpu")
        self.to(self.device)
        
        # Topic/Aspect Distribution Priors
        self.delta_prior = delta_prior if delta_prior else dist.Normal(0, delta_scale * torch.ones(num_topics, device=self.device))
        self.tau_prior   = tau_prior if tau_prior else dist.Normal(0, tau_scale * torch.ones(num_aspects, device=self.device))
        
        # Sigma level Distribution
        self.sigma_0, self.sigma_1 = sigma_0, sigma_1
        self.sigma_prior = sigma_prior if sigma_prior else dist.Beta(self.sigma_0, self.sigma_1)
        
        # Psi Prior Distributions
        
        self.gamma_0, self.gamma_1 = gamma_0, gamma_1
        # Psi prior for background
        self.psi0_prior   = dist.Beta(self.gamma_0, self.gamma_1)
        # Psi prior for topics
        self.psi_prior    = dist.Beta(torch.ones(self.num_topics) * self.gamma_0, torch.ones(self.num_topics) * self.gamma_1)
        
        # Background Topic Distribution over Vocab
        self.bg_topic     = bg_topic if bg_topic is not None else self.init_bg_topic()
        self.bg_topic     = self.bg_topic.to(self.device)
    
    
    def init_bg_topic(self):
        inits = torch.zeros(self.vocab_size, requires_grad = True, device = self.device)
        inits = nn.init.uniform_(inits)
        return inits
        
    def model(self,
              bows: torch.Tensor, 
              num_docs: int):
        """
            The Topic-Aspect model to generate a corpus of documents
            
            Parameters:
                bows: torch.Tensor
                    The corpus of source document BOWS for loss calculations
                num_docs: int
                    The total number of documents in the source corpus
                h2: torch.Tensor (optional, default = None)
                    If half (either ordered or random) of the document texts are passed for
                        bows, then the remaining half should be passed as h2
        """
        
        pyro.param("bg_topic", self.bg_topic)
        pyro.module('top_decoder', self.top_decoder)
        pyro.module('asp_decoder', self.asp_decoder)
        pyro.module('ta_decoder', self.ta_decoder)        
        
        psi_0 = pyro.sample('psi0', self.psi0_prior).to(self.device)
        psi   = pyro.sample('psi', self.psi_prior.to_event(1)).to(self.device)     
        
        with pyro.plate("documents", num_docs, subsample = bows):
            delta = pyro.sample("delta", self.delta_prior.to_event(1))
            tau   = pyro.sample("tau", self.tau_prior.to_event(1))
            sigma = pyro.sample('sigma', self.sigma_prior).to(self.device)

            # Softmax to calculate theta and pi, the distribution over topics and aspects: (Docs, Topics), (Docs, Aspects)
            theta = F.softmax(delta, -1)
            pi    = F.softmax(tau, -1)

            # Decode the topic distribution to generate distribution over words: (Docs, Vocab Size)
            top_probs = torch.softmax(self.top_decoder(theta), -1)
            asp_probs = torch.softmax(self.asp_decoder(pi), -1)
            ta_probs  = torch.softmax(self.ta_decoder(theta, pi), -1)
            
            bg_psi  = psi_0 * (1.-sigma)
            top_psi = (theta @ psi.T) * sigma
            
            b_bg_topic = torch.softmax(self.bg_topic, -1).unsqueeze(0).repeat((top_probs.shape[0], 1))
            probs_l0 = ((b_bg_topic * (1-bg_psi).unsqueeze(1)) + (asp_probs * bg_psi.unsqueeze(1))) * (1-sigma).unsqueeze(1)
            probs_l1 = ((top_probs * (1-top_psi).unsqueeze(1)) + (ta_probs * top_psi.unsqueeze(1))) * sigma.unsqueeze(1)
            all_probs = probs_l0 + probs_l1

            # Maximum document length for multinomial distribution sampling of reconstruction
            total_count = int((bows).sum(-1).max()) 

            # Sample document reconstruction from multinomial characterized by decoded topic distribution
            pyro.sample(
                'words',
                dist.Multinomial(total_count, probs = all_probs),
                obs=bows
            )

    def guide(self, 
              bows: torch.Tensor, 
              num_docs: int, 
              h2: torch.Tensor = None):
        """
            The ProdLDA guide for learning latent variables
            
            Parameters:
                bows: torch.Tensor
                    The corpus of source document BOWS for loss calculations
                num_docs: int
                    The total number of documents in the source corpus
                h2: torch.Tensor (optional, default = None)
                    If half (either ordered or random) of the document texts are passed for
                        bows, then the remaining half should be passed as h2
        """
        
        pyro.module('delta_encoder', self.delta_encoder)
        pyro.module('tau_encoder', self.tau_encoder)
        
        psi_0 = pyro.sample('psi0', self.psi0_prior).to(self.device)
        psi   = pyro.sample('psi', self.psi_prior.to_event(1)).to(self.device)

        # document plate
        with pyro.plate("documents", num_docs, subsample = bows):

            delta_loc, delta_sigma = self.delta_encoder(bows.float())
            tau_loc, tau_sigma     = self.tau_encoder(bows.float())
            

            pyro.sample(f"delta", dist.Normal(delta_loc, delta_sigma).to_event(1))
            pyro.sample(f"tau",   dist.Normal(tau_loc, tau_sigma).to_event(1))
            sigma = pyro.sample('sigma', self.sigma_prior).to(self.device)

    def betas(self):
        top_beta = self.top_decoder.beta.weight.detach().T
        asp_beta = self.asp_decoder.beta.weight.detach().T
        ta_beta  = self.ta_decoder.beta.weight.detach().T
        bg_beta  = self.bg_topic.detach()
        
        return bg_beta, top_beta, asp_beta, ta_beta
    
    def reconstruct_doc(self, bow, num_particles = 50):
        self.eval()

        num_docs = bow.shape[0]

        with torch.no_grad():
            delta_loc, delta_scale = self.delta_encoder(bow)
            tau_loc, tau_scale = self.tau_encoder(bow)
            
        delta_samples = dist.Normal(delta_loc, delta_scale).sample((num_particles,))
        tau_samples   = dist.Normal(tau_loc, tau_scale).sample((num_particles,))
        sigma         = self.sigma_prior.sample((num_particles,))

        theta = F.softmax(delta, -1)
        pi    = F.softmax(tau, -1)

        # decode for reconstruction
        with torch.no_grad():
            psi_0 = self.psi0_prior.sample()
            psi   = self.psi_prior.sample()  
        
            theta = theta.view(num_docs*num_particles, -1)
            pi = pi.view(num_docs*num_particles, -1)            
            
            top_probs = torch.softmax(self.top_decoder(theta), -1)
            asp_probs = torch.softmax(self.asp_decoder(pi), -1)
            ta_probs  = torch.softmax(self.ta_decoder(theta, pi), -1)
            
            bg_psi  = psi_0 * (1.-sigma)
            top_psi = (theta @ psi.T) * sigma
            
            b_bg_topic = torch.softmax(self.bg_topic, -1).unsqueeze(0).repeat((top_probs.shape[0], 1))
            probs_l0 = ((b_bg_topic * (1-bg_psi).unsqueeze(1)) + (asp_probs * bg_psi.unsqueeze(1))) * (1-sigma).unsqueeze(1)
            probs_l1 = ((top_probs * (1-top_psi).unsqueeze(1)) + (ta_probs * top_psi.unsqueeze(1))) * sigma.unsqueeze(1)
            all_probs = probs_l0 + probs_l1

        return all_probs
    
    def calc_perplexity(self, test_half_loader, num_particles = 50):
        """
        Calculate perplexity
        """

        # Accumulated perplexity
        total_ce = 0
        total_num_words = 0

        for i, batch in enumerate(test_half_loader):   
            bow = batch['bow'].to(self.device)

            bow_recon = self.reconstruct_doc(bow, num_particles = num_particles)
            ces       = (-bow*torch.log(bow_recon))
            total_ce += ces.sum().cpu().item()
            total_num_words += bow.sum()
            
        ce = total_ce / total_num_words
        perp = torch.exp(ce)

        return perp, ce
    
    def pred_aspect_dists(self, bow):
        with torch.no_grad():
            delta_loc, delta_scale = self.delta_encoder(bow)
            tau_loc, tau_scale = self.tau_encoder(bow)

            theta = delta_loc.softmax(dim = -1)
            pi    = tau_loc.softmax(dim = -1)

            topic_idx = theta.argmax(dim = -1)

            ta_beta = self.ta_decoder.beta.weight.detach().T
            asp1 = ta_beta[topic_idx].softmax(dim = -1)
            asp2 = ta_beta[topic_idx + ta_beta.shape[0] // 2].softmax(dim = -1)
        
        return asp1, asp2
        
    
    def save(self, save_path:str):
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        torch.save(self.delta_encoder.state_dict(), f'{save_path}/delta_encoder.pt')
        torch.save(self.tau_encoder.state_dict(), f'{save_path}/tau_encoder.pt')
        torch.save(self.top_decoder.state_dict(), f'{save_path}/top_decoder.pt')
        torch.save(self.asp_decoder.state_dict(), f'{save_path}/asp_decoder.pt')
        torch.save(self.ta_decoder.state_dict(), f'{save_path}/ta_decoder.pt')
        
        with open(f'{save_path}/delta_prior.pkl', 'wb') as f:
            pkl.dump(self.delta_prior, f)
            
        with open(f'{save_path}/psi0_prior.pkl', 'wb') as f:
            pkl.dump(self.psi0_prior, f)
            
        with open(f'{save_path}/psi_prior.pkl', 'wb') as f:
            pkl.dump(self.psi_prior, f)
        
        with open(f'{save_path}/bg_topic.pkl', 'wb') as f:
            pkl.dump(self.bg_topic, f)
        
        config_dict = {
            'model_type':  'topic_aspect_model',
            'num_topics':  self.num_topics,
            'num_aspects': self.num_aspects,
            'hidden_size': self.hidden_size,
            'vocab_size':  self.vocab_size,
            'dropout':     self.dropout,
            'gamma_0':     self.gamma_0,
            'gamma_1':     self.gamma_1,
            'sigma_0':     self.sigma_0,
            'sigma_1':     self.sigma_1
        }
        
        with open(f'{save_path}/config.json', 'w') as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, model_path, device = None):
        
        with open(f'{model_path}/config.json', 'r') as f:
            config_dict = json.load(f)
        
        if config_dict['model_type'] != 'topic_aspect_model':
            raise Exception(f'Pretrained model of type {config_dict["model_type"]} is not of type ProdLDA')
        
        num_topics = config_dict['num_topics']
        num_aspects = config_dict['num_aspects']
        
        GS_DEFAULT = 10.
        gamma_0, gamma_1 = config_dict['gamma_0'] if 'gamma_0' in config_dict.keys() else GS_DEFAULT, config_dict['gamma_1'] if 'gamma_1' in config_dict.keys() else GS_DEFAULT
        sigma_0, sigma_1 = config_dict['sigma_0'] if 'sigma_0' in config_dict.keys() else GS_DEFAULT, config_dict['sigma_1'] if 'sigma_1' in config_dict.keys() else GS_DEFAULT
        
        delta_encoder = Encoder(config_dict['vocab_size'],
                         config_dict['num_topics'],
                         config_dict['hidden_size'],
                         config_dict['dropout'])
        
        tau_encoder = Encoder(config_dict['vocab_size'],
                         config_dict['num_aspects'],
                         config_dict['hidden_size'],
                         config_dict['dropout'])
        
        top_decoder = Decoder(config_dict['vocab_size'],
                         config_dict['num_topics'],
                         config_dict['dropout'])
        
        asp_decoder = Decoder(config_dict['vocab_size'],
                         config_dict['num_aspects'],
                         config_dict['dropout'])
        
        ta_decoder = JointDecoder(config_dict['vocab_size'],
                         config_dict['num_topics'],
                         config_dict['num_aspects'],
                         config_dict['dropout'])
        
        delta_encoder.load_state_dict(torch.load(f'{model_path}/delta_encoder.pt'))
        tau_encoder.load_state_dict(torch.load(f'{model_path}/tau_encoder.pt'))
        top_decoder.load_state_dict(torch.load(f'{model_path}/top_decoder.pt'))
        asp_decoder.load_state_dict(torch.load(f'{model_path}/asp_decoder.pt'))
        ta_decoder.load_state_dict(torch.load(f'{model_path}/ta_decoder.pt'))
        
        with open(f'{model_path}/delta_prior.pkl', 'rb') as f:
            delta_prior = pkl.load(f)
            
        with open(f'{model_path}/psi0_prior.pkl', 'rb') as f:
            psi0_prior = pkl.load(f)
            
        with open(f'{model_path}/psi_prior.pkl', 'rb') as f:
            psi_prior = pkl.load(f)
            
        with open(f'{model_path}/bg_topic.pkl', 'rb') as f:
            bg_topic = pkl.load(f)
                        
        model = cls(num_topics, num_aspects,
                    delta_encoder = delta_encoder, tau_encoder = tau_encoder, 
                    top_decoder = top_decoder, asp_decoder = asp_decoder, ta_decoder = ta_decoder,
                    bg_topic = bg_topic,
                    delta_prior = delta_prior, 
                    gamma_0 = gamma_0, gamma_1 = gamma_1,
                    sigma_0 = sigma_0, sigma_1 = sigma_1,
                    device = device)
        
        return model