import os
import sys
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

class PositionalEncoding:
    def __init__(self, num_dimensions = 32, pos = 0):
        self.num_dimensions = num_dimensions
        self.pos = pos
    
    def encodingStep(self):
        encoding = np.zeros(self.num_dimensions)
        for i in range(self.num_dimensions):
            if i % 2 == 0:
                encoding[i] = np.sin(self.pos / (10000 ** (2*i / self.num_dimensions)))
            else:
                encoding[i] = np.cos(self.pos / (10000 ** (2*i / self.num_dimensions)))

        return encoding

class AttentionHead(nn.Module):
    def __init__(self, input_dimensions, num_heads):
        super(AttentionHead, self).__init__()
        self.input_dimensions = input_dimensions
        self.num_heads = num_heads
    
    def attentionCalculation(self):
        query = np.zeros((self.num_heads, self.input_dimensions))
        key = np.zeros((self.num_heads, self.input_dimensions))
        value = np.zeros((self.num_heads, self.input_dimensions))

    

class Encoder:
    def __init__(self):

class Decoder:
    def __init__(self):

class TransformerBlock:
    def __init__(self, nn.module):

class PredictionModel:
    def __init__(self, transformermodel, model_path="",):

    def predict():
        self.current_seq_ix = None
        self.sequence_history = []
        
        # Define a simple GRU model
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)  # Output layer for t0 and t1

        # Initialize hidden state
        self.hidden = None