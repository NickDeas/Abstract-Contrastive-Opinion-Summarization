import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm
import torch.nn.functional as F

from typing import Optional, List, Union, Tuple

# ----- Encoder -----

class Encoder(nn.Module):
    """
        Base class for the document encoder used within the guide
    """
    
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        
        super().__init__()
        
        self.vocab_size  = vocab_size
        self.num_topics  = num_topics
        self.hidden_size = hidden
        self.dropout     = dropout
        
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        
        self.fc2 = nn.Linear(hidden, hidden)
            
        self.fcmu = nn.Linear(hidden, num_topics)
            
        self.fclv = nn.Linear(hidden, num_topics)
        
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
         
        h = F.softplus(self.fc2(h))
        
        h = self.drop(h)
        
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
            
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = 1.0e-10 + (0.5 * logtheta_logvar).exp()  # Defenisvely enforces positivity
            
        return logtheta_loc, logtheta_scale

# ----- Decoders -----

class Decoder(nn.Module):
    """
        Base class for the document decoder used in the model
    """
    
    def __init__(self, vocab_size, num_topics, dropout):
        
        super().__init__()
        
        self.vocab_size  = vocab_size
        self.num_topics  = num_topics
        self.dropout     = dropout
        
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return self.bn(self.beta(inputs))
    
    
class JointDecoder(nn.Module):
    """
        Decoder that jointly decodes word probabilities based on both topics and aspects
    """
    
    def __init__(self, vocab_size, num_topics, num_aspects, dropout):
        
        super().__init__()
        
        self.vocab_size  = vocab_size
        self.num_topics  = num_topics
        self.num_aspects = num_aspects
        self.dropout     = dropout
        
        self.beta = nn.Linear(num_topics*num_aspects, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, top_inputs, asp_inputs):        
        top_inputs = top_inputs.repeat((1, self.num_aspects))
        asp_inputs = asp_inputs.unsqueeze(2).repeat((1, 1, self.num_topics)).reshape((asp_inputs.shape[0], -1, 2*self.num_topics)).squeeze()
        inputs = top_inputs * asp_inputs
        
        inputs = self.drop(inputs)
        
        return self.bn(self.beta(inputs))