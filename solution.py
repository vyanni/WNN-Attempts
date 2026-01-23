import os
import sys

from example_solution.utils import DataPoint, ScorerStepByStep

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import pyarrow as pyarrow

class PositionalEncoding:
    def __init__(self, dimensionSize = 32, timeLength = 100):
        self.dimensionSize = dimensionSize
        self.timeLength = timeLength

        self.encodingVector = self.encodingStep()
        #Takes in the dimension size of 32 for the market state vector, 
        #along with how far back in data which the transformer takes for the time-series, 
        #currently 100 steps since the first 0-99 steps are for training.
    
    def encodingStep(self):
        self.encodingVector = np.zeros(self.timeLength, self.dimensionSize)
        currentPosition = np.arange(0, self.timeLength)[:, np.newaxis]
        divisiveTerm = np.arange(0, self.dimensionSize) * -(10000 / self.dimensionSize)
        #Starts the original and only encoding vector which will be added to the input
        #Starts the position as an array for all the timesteps, then uses "[:, np.newaxis]" to turn it into a column vector
        #[:, np.newaxis] turns it into a 2D array of the original size of the array, by 1

        self.encodingVector[:, 0::2] = np.sin(currentPosition / divisiveTerm)
        self.encodingVector[:, 1::2] = np.cos(currentPosition / divisiveTerm)

        return torch.tensor(self.encodingVector, torch.float32)

    def getEncodingVector(self):
        return self.encodingVector

class AttentionHead(nn.Module):
    def __init__(self, dimensionSize, attentionHeadOutputDimension):
        super(AttentionHead, self).__init__()
        self.dimensionSize = dimensionSize
        self.attentionHeadOutputDimension = attentionHeadOutputDimension
        #The attention head calculates the key, query, and value vectors for an encoded input vector
        #These are done just by doing a linear multiplication with qkv weight matrices

        self.queryWeights = nn.Linear(self.dimensionSize, self.attentionHeadOutputDimension)
        self.keyWeights = nn.Linear(self.dimensionSize, self.attentionHeadOutputDimension)
        self.valueWeights = nn.Linear(self.dimensionSize, self.attentionHeadOutputDimension)

    def attentionCalculation(self, marketState):
        query = self.queryWeights(marketState)
        key = self.keyWeights(marketState)
        value = self.valueWeights(marketState)

        attentionFunction = torch.matmul(query, key.transpose(-2, self.dimensionSize)) / np.sqrt(self.dimensionSize)
        attentionOutput = F.softmax(attentionFunction, dim=-1)

        output = torch.matmul(attentionOutput, value)
        return output
    
class MultiheadAttention(nn.Module):
    def __init__(self, numHeads, dimensionSize, attentionHeadOutputDimension):
        super(MultiheadAttention, self).__init__()
        self.dimensionSize = dimensionSize
        self.numHeads = numHeads
        self.attentionHeadOutputDimension = attentionHeadOutputDimension

        self.attention_heads = nn.ModuleList([AttentionHead(self.dimensionSize, self.attentionHeadOutputDimension)] for i in numHeads)

        self.final_linear = nn.Linear(numHeads * attentionHeadOutputDimension, attentionHeadOutputDimension)

    def forwardAttention(self, marketState):
        outputArray = [head(marketState) for head in self.numHeads]
        
        concatenatedHeads = torch.Cat(outputArray, dim=-1)
        finalOutput = self.final_linear(concatenatedHeads)

        return finalOutput


class Encoder(nn.Module):
    def __init__(self, numHeads, dimensionSize, attentionHeadOutputDimension, feedforward_dimensions):
        super(Encoder, self).__init__()
        self.numHeads = numHeads
        self.dimensionSize = dimensionSize
        self.attentionHeadOutputDimension = attentionHeadOutputDimension
        self.feedForward_dimensions = feedforward_dimensions

        self.attention = MultiheadAttention(self.numHeads, self.dimensionSize, self.attentionHeadOutputDimension)

        self.feedForward = nn.Sequential(
            nn.Linear(attentionHeadOutputDimension, feedforward_dimensions)
            nn.ReLU()
            nn.Linear(feedforward_dimensions, attentionHeadOutputDimension)
        )

        self.normalizationLayer1 = nn.LayerNorm(attentionHeadOutputDimension)
        self.normalizationLayer2 = nn.LayerNorm(attentionHeadOutputDimension)

    def forwardEncoding(self, marketState):
        attentionOutput = self.attention(marketState)
        attentionOutput = self.normalizationLayer1(attentionOutput + marketState)

        finalAttention = self.feedForward(attentionOutput)
        finalOutput = self.normalizationLayer2(finalAttention + attentionOutput)
        return finalOutput

class TransformerBlock(nn.Module):
    def __init__(self, numLayers, numHeads, dimensionSize, attentionHeadOutputDimension, feedforward_dimensions):
        super(TransformerBlock, self).__init__()
        self.numLayers = numLayers
        self.encoderLayers = nn.ModuleList([
            Encoder(numHeads, dimensionSize, attentionHeadOutputDimension, feedforward_dimensions) 
            for i in range(numLayers)
        ])

    def forward(self, marketState):
        for i in range(self.numLayers):
            marketState = self.encoderLayers[i](marketState)
        
        return marketState


class PredictionModel:
    def __init__(self, transformermodel, model_path="",):
        self.current_seq_ix = None
        self.sequence_history = []

        self.dimensionCompressor = nn.Linear(32, 2)

    def training(self, currentSeq: DataPoint):


    def predict(self, currentSeq: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = []
        
        self.sequence_history.append(currentSeq.state.copy())

        if not currentSeq.need_prediction:
            return None
        
        currentTransformer = TransformerBlock(8, 8, 32, 32, 32)

        transformerOutput = currentTransformer(currentSeq)
        finalPrediction = self.dimensionCompressor(transformerOutput)

        lossFunction = nn.L1Loss()
        lossValue = lossFunction(finalPrediction, currentSeq.state)
        lossValue.backward()

        finalLinear = nn.Linear(32, 2)
        prediction = finalLinear(finalPrediction)

        return prediction