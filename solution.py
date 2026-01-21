import os
import sys
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

class AttentionHead:
    def __init__(self):

class TransformerBlock:
    def __init__(self, nn.module):

class PositionalEncoding:
    def __init__(self):

class Encoder:
    def __init__(self):

class Decoder:
    def __init__(self):

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