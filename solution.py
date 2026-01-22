import os
import sys
from example_solution.utils import DataPoint
import torch as torch
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
    def __init__(self, input_dimensions, output_dimensions):
        super(AttentionHead, self).__init__()
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

        self.queryWeights = nn.Linear(self.input_dimensions, self.output_dimensions)
        self.keyWeights = nn.Linear(self.input_dimensions, self.output_dimensions)
        self.valueWeights = nn.Linear(self.input_dimensions, self.output_dimensions)

    def attentionCalculation(self, marketState):
        query = torch.matmul(self.queryWeights, marketState)
        key = torch.matmul(self.keyWeights, marketState)
        value = torch.matmul(self.valueWeights, marketState)

        attention_input = torch.matmul(query, key.transpose(-2, self.input_dimensions)) / np.sqrt(self.input_dimensions)
        final_attentionValue = F.softmax(attention_input, dim=-1)

        output = torch.matmul(final_attentionValue, value)

        return output
    
class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, input_dimensions, output_dimensions):
        super(MultiheadAttention, self).__init__()
        self.input_dimensions = input_dimensions
        self.num_heads = num_heads
        self.output_dimensions = output_dimensions

        self.attention_heads = nn.ModuleList([AttentionHead(self.input_dimensions, self.output_dimensions)] for i in num_heads)

        self.final_linear = nn.Linear(num_heads * output_dimensions, output_dimensions)

    def forwardAttention(self, marketState):
        outputArray = [head(marketState) for head in self.num_heads]
        
        concatenatedHeads = torch.Cat(outputArray, dim=-1)
        finalOutput = self.final_linear(concatenatedHeads)

        return finalOutput


class Encoder(nn.Module):
    def __init__(self, num_heads, input_dimensions, output_dimensions, feedforward_dimensions):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.feedForward_dimensions = feedforward_dimensions

        self.attention = MultiheadAttention(self.num_heads, self.input_dimensions, self.output_dimensions)

        self.feedForward = nn.Sequential(
            nn.Linear(output_dimensions, feedforward_dimensions)
            nn.ReLU()
            nn.Linear(feedforward_dimensions, output_dimensions)
        )
        self.normalizationLayer1 = nn.LayerNorm(output_dimensions)
        self.normalizationLayer2 = nn.LayerNorm(output_dimensions)

    def forwardEncoding(self, marketState):
        attentionOutput = self.attention(marketState)
        attentionOutput = self.normalizationLayer1(attentionOutput + marketState)

        finalAttention = self.feedForward(attentionOutput)
        finalOutput = self.normalizationLayer2(finalAttention + attentionOutput)
        return finalOutput

class Decoder:
    def __init__(self):


class TransformerBlock(nn.Module):
    def __init__(self, numLayers, num_heads, input_dimensions, output_dimensions, feedforward_dimensions):
        super(TransformerBlock, self).__init__()
        self.numLayers = numLayers
        self.encoderLayers = nn.ModuleList([Encoder(num_heads, input_dimensions, output_dimensions, feedforward_dimensions)] for i in range numLayers)

    def forward(self, marketState):
        encoderOutput = marketState

        for i in range self.numLayers:
            encoderOutput = self.encoderLayers[i](encoderOutput)
        
        return encoderOutput


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