import os
import sys
from example_solution.utils import DataPoint
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
    def __init__(self, input_dimensions):
        super(AttentionHead, self).__init__()
        self.input_dimensions = input_dimensions

        self.queryWeights = np.random.rand(self.num_heads, self.input_dimensions)
        self.keyWeights = np.random.rand(self.num_heads, self.input_dimensions)
        self.valueWeights = np.random.rand(self.num_heads, self.input_dimensions)

    def attentionCalculation(self, input):
        query = self.queryWeights.dot(input)
        key = self.keyWeights.dot(input)
        value = self.valueWeights.dot(input)

        attention_input = np.dot(query, key.T) / np.sqrt(self.input_dimensions)
        final_attentionValue = F.softmax(attention_input, dim=-1).dot(value)

        return final_attentionValue

class Encoder:
    def __init__(self, num_heads, final_input, input_dimensions):
        self.num_heads = num_heads
        self.final_input = final_input
        self.input_dimensions = input_dimensions

    def forwardEncoding(self, input):
        attention_heads = [AttentionHead(self.input_dimensions) for _ in range(self.num_heads)]
        head_outputs = [head.attentionCalculation(input) for head in attention_heads]
        concatenated_heads = np.concatenate(head_outputs, axis=-1)

        residual_connections = concatenated_heads + input

        feedForward = nn.ReLU(residual_connections)
        return feedForward

class Decoder:
    def __init__(self):

class TransformerBlock(nn.module):
    def __init__(self, num_layers, num_heads):
        super(AttentionHead, self).__init__()

        
        #layers = [Encoder(self. )]


class PredictionModel:
    def __init__(self, transformermodel, model_path="",):

    def predict(self, seq_dataPoint: DataPoint) -> np.ndarray | None:

        if not seq_dataPoint.need_prediction:
            return None
        
        self.current_seq_ix = None
        self.sequence_history = []
        
        # Define a simple GRU model
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)  # Output layer for t0 and t1

        # Initialize hidden state
        self.hidden = None

        prediction = np.zeros(2) # Placeholder for prediction output
        return prediction