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

        self.adaptivePositionAttention = nn.Sequential(
            nn.Linear(dimensionSize, dimensionSize),
            nn.ReLU(),
            nn.Linear(dimensionSize, dimensionSize),
            nn.Sigmoid()
        )

        #Takes in the dimension size of 32 for the market state vector, 
        #along with how far back in data which the transformer takes for the time-series, 
        #currently 100 steps since the first 0-99 steps are for training.
    
    def sinusodialEncoding(self):
        sinsusodialEncoding = torch.empty(self.timeLength, self.dimensionSize)
        currentPosition = torch.arange(0, self.timeLength)[:, torch.newaxis]
        divisiveTerm = torch.exp(
            torch.arange(0, self.dimensionSize, 2).float() * (-np.log(10000.0) / self.dimensionSize)
        )
        #Starts the original and only encoding vector which will be added to the input
        #Starts the position as an array for all the timesteps, then uses "[:, np.newaxis]" to turn it into a column vector
        #[:, np.newaxis] turns it into a 2D array of the original size of the array, by 1

        sinsusodialEncoding[:, 0::2] = torch.sin(currentPosition / divisiveTerm)
        sinsusodialEncoding[:, 1::2] = torch.cos(currentPosition / divisiveTerm) 

        self.sinsusodialWeight = nn.Parameter(torch.randn(self.timeLength, self.dimensionSize))

        return sinsusodialEncoding.float()
    
    def learnableEncoding(self):
        learnableEncoding = nn.Parameter(torch.randn(self.timeLength, self.dimensionSize) * 0.05)

        self.learnableWeight = nn.Parameter(torch.randn(self.timeLength, self.dimensionSize))

        return learnableEncoding.float()
        
    def getEncodingVector(self, marketStateBatch):
        positionalWeights = self.adaptivePositionAttention(marketStateBatch)

        encodingVector = (
            ((self.sinusodialEncoding() * self.sinsusodialWeight) + 
            (self.learnableEncoding() * self.learnableWeight)) * 
            (positionalWeights)
        )

        encodingVector = (marketStateBatch + encodingVector)
        return encodingVector
    
class FeatureAttention(nn.Module):
    def __init__(self, dimensionSize, extraFeatures):
        super(FeatureAttention, self).__init__()

        self.individualFeatureAttention = nn.Sequential(
            nn.Linear(dimensionSize, extraFeatures),
            nn.ReLU(),
            nn.Linear(extraFeatures, dimensionSize)
        )

class AttentionHead(nn.Module):
    def __init__(self, dimensionSize, attentionHeadOutputDimension, maxLength = 100):
        super(AttentionHead, self).__init__()
        self.dimensionSize = dimensionSize
        self.attentionHeadOutputDimension = attentionHeadOutputDimension
        #The attention head calculates the key, query, and value vectors for an encoded input vector
        #These are done just by doing a linear multiplication with qkv weight matrices

        self.queryWeights = nn.Linear(self.dimensionSize, self.attentionHeadOutputDimension)
        self.keyWeights = nn.Linear(self.dimensionSize, self.attentionHeadOutputDimension)
        self.valueWeights = nn.Linear(self.dimensionSize, self.attentionHeadOutputDimension)
        
        #Create upper matrix mask for the top right, preventing the transformer from taking into account a token's relation to a future token 
        self.matriceMask = torch.tril(torch.ones(maxLength, maxLength))
        #self.register_buffer("causal_mask", causal_mask)

    def forward(self, marketStateBatch):
        seqLength = marketStateBatch.shape[0]
        query = self.queryWeights(marketStateBatch)
        key = self.keyWeights(marketStateBatch)
        value = self.valueWeights(marketStateBatch)
        #It takes in the 100x32 vector, where its 32 dimensions by 100 tokens, then multiplies each token in each row
        #By the query, key, and value weights where it comes out as a 100x32 vector, same size etc    

        attentionScores = torch.matmul(query, torch.transpose(key, 0, 1)) / np.sqrt(self.dimensionSize)
        
        mask = self.matriceMask[:seqLength, :seqLength]
        attentionScores = attentionScores.masked_fill(mask == 0, float('-inf'))
        
        attentionOutput = F.softmax(attentionScores, dim=-1)
        # It then takes the query, multiplies it by the transpose of the key vector (matrix with all the tokens together)
        # It divides it by the square root of the dimensions just to keep it from bloating up, then softmaxes it
        
        # Handle NaN from softmax of all -inf
        attentionOutput = torch.nan_to_num(attentionOutput, 0.0)

        output = torch.matmul(attentionOutput, value)
        #Dot products the softmax and the value in order to get the output, another 100x32 matrix

        return output
    
class MultiheadAttention(nn.Module):
    def __init__(self, numHeads, dimensionSize, attentionHeadOutputDimension, maxLength = 100):
        super(MultiheadAttention, self).__init__()
        self.dimensionSize = dimensionSize
        self.numHeads = numHeads
        self.attentionHeadOutputDimension = attentionHeadOutputDimension

        self.attentionHeads = nn.ModuleList([AttentionHead(self.dimensionSize, self.attentionHeadOutputDimension, maxLength = 100) for i in range(numHeads)])
        self.finalLinear = nn.Linear(numHeads * attentionHeadOutputDimension, attentionHeadOutputDimension)

    def forward(self, marketStateBatch):
        outputArray = [head(marketStateBatch) for head in self.attentionHeads]
        #Calculates the attention for each head, will probably do around 8

        concatenatedHeads = torch.cat(outputArray, dim=-1)
        finalOutput = self.finalLinear(concatenatedHeads)
        #Concatenates it, so there's an array of 100x32 matrices, takes each one and pushes them together to be a
        #100x(32*number of heads) long matrix, then another linear transformation brings it back to 100x32 matrix
        #So this still returns a 100x32 matrix

        return finalOutput

class Encoder(nn.Module):
    def __init__(self, numHeads, dimensionSize, attentionHeadOutputDimension, feedforward_dimensions):
        super(Encoder, self).__init__()
        self.numHeads = numHeads
        self.dimensionSize = dimensionSize
        self.attentionHeadOutputDimension = attentionHeadOutputDimension
        self.feedForward_dimensions = feedforward_dimensions

        self.attention = MultiheadAttention(self.numHeads, self.dimensionSize, self.attentionHeadOutputDimension, maxLength = 100)

        self.feedForward = nn.Sequential(
            nn.Linear(attentionHeadOutputDimension, feedforward_dimensions),
            nn.ReLU(),
            nn.Linear(feedforward_dimensions, attentionHeadOutputDimension)
        )

        self.normalizationLayer1 = nn.LayerNorm(attentionHeadOutputDimension)
        self.normalizationLayer2 = nn.LayerNorm(attentionHeadOutputDimension)

    def forward(self, marketStateBatch):
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