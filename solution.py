import os
import sys

from example_solution.utils import DataPoint, ScorerStepByStep

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import pyarrow as pyarrow

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

class PositionalEncoding:
    def __init__(self, dimensionSize = 32, timeLength = 100):
        self.dimensionSize = dimensionSize
        self.timeLength = timeLength

        self.encodingVector = self.encodingStep()
        #Takes in the dimension size of 32 for the market state vector, 
        #along with how far back in data which the transformer takes for the time-series, 
        #currently 100 steps since the first 0-99 steps are for training.
    
    def encodingStep(self):
        self.encodingVector = torch.empty(self.timeLength, self.dimensionSize)
        currentPosition = torch.arange(0, self.timeLength)[:, np.newaxis]
        divisiveTerm = torch.arange(0, self.dimensionSize) * -(10000 / self.dimensionSize)
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

    def attentionCalculation(self, marketStateBatch):
        query = self.queryWeights(marketStateBatch)
        key = self.keyWeights(marketStateBatch)
        value = self.valueWeights(marketStateBatch)
        #It takes in the 32x100 vector, where its 32 dimensions by 100 tokens    

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

        self.attentionHeads = nn.ModuleList([AttentionHead(self.dimensionSize, self.attentionHeadOutputDimension) for i in range(numHeads)])

        self.final_linear = nn.Linear(numHeads * attentionHeadOutputDimension, attentionHeadOutputDimension)

    def forwardAttention(self, marketStateBatch):
        outputArray = [head(marketStateBatch) for head in self.attentionHeads]
        
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
            nn.Linear(attentionHeadOutputDimension, feedforward_dimensions),
            nn.ReLU(),
            nn.Linear(feedforward_dimensions, attentionHeadOutputDimension)
        )

        self.normalizationLayer1 = nn.LayerNorm(attentionHeadOutputDimension)
        self.normalizationLayer2 = nn.LayerNorm(attentionHeadOutputDimension)

    def forwardEncoding(self, marketStateBatch):
        attentionOutput = self.attention(marketStateBatch)
        attentionOutput = self.normalizationLayer1(attentionOutput + marketStateBatch)

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

    def forward(self, marketStateBatch):
        for i in range(self.numLayers):
            marketStateBatch = self.encoderLayers[i](marketStateBatch)
        
        return marketStateBatch


class PredictionModel:
    def __init__(self, model_path="",):
        self.current_seq_ix = None
        self.sequence_history = []

        self.dimensionCompressor = nn.Linear(32, 2)
        self.currentTransformer = TransformerBlock(numLayers=8, numHeads=8, dimensionSize=32, attentionHeadOutputDimension=32, feedforward_dimensions=32)

        self.PositionalEncodingObject = PositionalEncoding(dimensionSize = 32, timeLength = 100)
        self.positionalEncodingVector = self.PositionalEncodingObject.getEncodingVector()

        self.lossFunction = nn.MSELoss()
        self.optimizerFunction = torch.optim.Adam(currentTransformer.parameters(), lr=0.001)

    def training(self, currentSeq: DataPoint):
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty()
        
        self.sequence_history = torch.cat([self.sequence_history, currentSeq.state.copy()], dim = 1)

        if not currentSeq.need_prediction:
            return None

        self.sequence_history = self.sequence_history + self.positionalEncodingVector

        currentTransformer = TransformerBlock(numLayers=8, numHeads=8, dimensionSize=32, attentionHeadOutputDimension=32, feedforward_dimensions=32)

        transformerOutput = currentTransformer(currentSeq)
        finalPrediction = self.dimensionCompressor(transformerOutput)

        finalLinear = nn.Linear(32, 2)
        prediction = finalLinear(finalPrediction)

        self.lossValue = self.lossFunction(prediction, currentSeq.state)
        self.lossValue.backward()

    def predict(self, currentSeq: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty()
        
        self.sequence_history = torch.cat([self.sequence_history, currentSeq.state.copy()], dim = 1)

        if not currentSeq.need_prediction:
            return None
        
        inputTokens = self.sequence_history[-100:, :] + self.positionalEncodingVector 
        #Both are vector of size 32 x 100 for the 32 dimension market state, by the past 100 time steps
        #The transformer always takes in a 32x100 vector, where each row of 32x1 is 1 token

        transformerOutput = self.currentTransformer(inputTokens)
        finalPrediction = self.dimensionCompressor(transformerOutput)

        finalLinear = nn.Linear(32, 2)
        prediction = finalLinear(finalPrediction)

        return prediction

trainingFileDirectory = f"{CURRENT_DIR}/../datasets/valid.parquet"
df = pd.read_parquet(trainingFileDirectory)
print(df.head())