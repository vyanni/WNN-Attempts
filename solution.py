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

        return self.encodingVector.float()

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
        #

    def attentionCalculation(self, marketStateBatch):
        query = self.queryWeights(marketStateBatch)
        key = self.keyWeights(marketStateBatch)
        value = self.valueWeights(marketStateBatch)
        #It takes in the 100x32 vector, where its 32 dimensions by 100 tokens, then multiplies each token in each row
        #By the query, key, and value weights where it comes out as a 100x32 vector, same size etc    

        attentionFunction = torch.matmul(query, torch.transpose(key, 0, 1)) / np.sqrt(self.dimensionSize)
        attentionOutput = F.softmax(attentionFunction, dim=-1)
        # It then takes the query, multiplies it by the transpose of the key vector (matrix with all the tokens together)
        # It divides it by the square root of the dimensions just to keep it from bloating up, then softmaxes it

        output = torch.matmul(attentionOutput, value)
        #Dot products the softmax and the value in order to get the output, another 100x32 matrix

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
        #Calculates the attention for each head, will probably do around 8

        concatenatedHeads = torch.cat(outputArray, dim=-1)
        finalOutput = self.final_linear(concatenatedHeads)
        #Concatenates it, so there's an array of 100x32 matrices, takes each one and pushes them together to be a
        # 100x(32*number of heads) long matrix, then another linear transformation brings it back to 100x32 matrix
        #So this still returns a 100x32 matrix

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
            Encoder(numHeads, dimensionSize, attentionHeadOutputDimension, feedforward_dimensions) for i in range(numLayers)
        ])

    def forward(self, marketStateBatch):
        for i in range(self.numLayers):
            marketStateBatch = self.encoderLayers[i](marketStateBatch)
        
        return marketStateBatch

class PredictionModel:
    def __init__(self, model_path="",):
        self.current_seq_ix = None
        self.sequence_history = []

        self.currentTransformer = TransformerBlock(numLayers=8, numHeads=8, dimensionSize=32, attentionHeadOutputDimension=32, feedforward_dimensions=32)

        self.finalLinear = nn.Linear(32, 2)
        #Brings it down from 1x32 to 1x2 

        self.PositionalEncodingObject = PositionalEncoding(dimensionSize = 32, timeLength = 100)
        self.positionalEncodingVector = self.PositionalEncodingObject.getEncodingVector()

        self.lossFunction = nn.MSELoss()
        self.optimizerFunction = torch.optim.Adam(self.currentTransformer.parameters(), lr=0.001)
        self.finalOptimizer = torch.optim.Adam(self.finalLinear.parameters(), lr=0.001)

    def training(self, currentSeq: DataPoint):
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty()
        
        self.sequence_history = torch.stack([self.sequence_history, currentSeq.state.copy()], dim = 1)

        if not currentSeq.need_prediction:
            return None

        inputTokens = self.sequence_history[-100:, :] + self.positionalEncodingVector
        
        transformerOutput = self.currentTransformer(inputTokens)
        #Goes through the whole transformer process, with attention etc, outputs 100x32 matrix

        singleTimeStep = torch.mean(transformerOutput, dim = 0)
        #Turns it into a 1x32 matrix for just a single timestep

        finalPrediction = self.finalLinear(singleTimeStep)
        #Changes it into a 1x2 vector for the predictions of t0 and t1
    
        prediction = finalPrediction.numpy()
    
        self.lossValue = self.lossFunction(prediction, currentSeq.state)
        self.lossValue.backward()

        self.optimizerFunction.step()
        self.optimizerFunction.zero_grad()

        self.finalOptimizer.step()
        self.finalOptimizer.zero_grad()

    def predict(self, currentSeq: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty()
        
        self.sequence_history = torch.cat([self.sequence_history, currentSeq.state.copy()], dim = 1)

        if not currentSeq.need_prediction:
            return None
        
        inputTokens = self.sequence_history[-100:, :] + self.positionalEncodingVector 
        #Both are vector of size 100x32 for the 32 dimension market state, by the past 100 time steps
        #The transformer always takes in a 100x32 vector, where each row of 1x32 is 1 token
        #Its declared as (rows, columns), but mathematically, its columns x rows

        transformerOutput = self.currentTransformer(inputTokens)
        #Goes through the whole transformer process, with attention etc, outputs 100x32 matrix

        singleTimeStep = torch.mean(transformerOutput, dim = 0)
        #Turns it into a 1x32 matrix for just a single timestep

        finalPrediction = self.finalLinear(singleTimeStep)
        #Changes it into a 1x2 vector for the predictions of t0 and t1
    

        prediction = finalPrediction.numpy()

        return prediction

trainingFileDirectory = f"{CURRENT_DIR}/../datasets/valid.parquet"
df = pd.read_parquet(trainingFileDirectory)
print(df.head())