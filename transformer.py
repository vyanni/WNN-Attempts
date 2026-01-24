import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import pyarrow as pyarrow

class PositionalEncoding(nn.Module):
    def __init__(self, dimensionSize = 32, timeLength = 100, dropout = 0.1):
        super().__init__()
        self.dimensionSize = dimensionSize
        self.timeLength = timeLength
        self.dropout = dropout

        self.adaptivePositionAttention = nn.Sequential(
            nn.Linear(dimensionSize, dimensionSize // 4),
            nn.ReLU(),
            nn.Linear(dimensionSize // 4, 1),
            nn.Sigmoid()
        )

        self.sinsusodialWeight = nn.Parameter(torch.ones(1))
        self.learnableWeight = nn.Parameter(torch.ones(1))

        #Takes in the dimension size of 32 for the market state vector, 
        #along with how far back in data which the transformer takes for the time-series, 
        #currently 100 steps since the first 0-99 steps are for training.
    
    def sinusodialEncoding(self):
        sinusodialEncoded = torch.zeros(self.timeLength, self.dimensionSize)
        currentPosition = torch.arange(self.timeLength).unsqueeze(1).float()
        divisiveTerm = torch.exp(
            torch.arange(0, self.dimensionSize, 2).float() * (-np.log(10000.0) / self.dimensionSize)
        )
        #self.register_buffer("sinEn", sinsusodialEncoding)

        #Starts the original and only encoding vector which will be added to the input
        #Starts the position as an array for all the timesteps, then uses "[:, np.newaxis]" to turn it into a column vector
        #[:, np.newaxis] turns it into a 2D array of the original size of the array, by 1

        sinusodialEncoded[:, 0::2] = torch.sin(currentPosition / divisiveTerm)
        sinusodialEncoded[:, 1::2] = torch.cos(currentPosition / divisiveTerm) 

        return sinusodialEncoded.float()
    
    def learnableEncoding(self):
        learnableEncoded = nn.Parameter(torch.randn(self.timeLength, self.dimensionSize) * 0.05)

        return learnableEncoded.float()
    
    def forward(self, marketStateBatch):
        self.sinusodiualEncoded = self.sinusodialEncoding()
        self.learnableEncoded = self.learnableEncoding()

        positionalWeights = self.adaptivePositionAttention(marketStateBatch)

        encodingVector = (
            ((self.sinusodiualEncoded * self.sinsusodialWeight) + 
            (self.learnableEncoded * self.learnableWeight)) * 
            (positionalWeights)
        )

        encodingVector = (marketStateBatch + encodingVector)
        return encodingVector
    
class FeatureAttention(nn.Module):
    def __init__(self, dimensionSize, hiddenDimension = 16):
        super().__init__()

        self.individualFeatureAttention = nn.Sequential(
            nn.Linear(dimensionSize, hiddenDimension),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hiddenDimension, dimensionSize),
            nn.Softmax(dim = -1)
        )

    def forward(self, marketStateBatch):
        importantFeatures = marketStateBatch * self.individualFeatureAttention(marketStateBatch)

        return importantFeatures

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

class HighwayNetwork(nn.Module):
    def __init__(self, dimensionSize, outputDimensions, num_layers= 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(nn.Linear(dimensionSize, dimensionSize))
            self.gates.append(nn.Linear(dimensionSize, dimensionSize))
        
        self.final = nn.Linear(dimensionSize, outputDimensions)
    
    def forward(self, marketStateBatch):
        for layer, gate in zip(self.layers, self.gates):
            carryGate = torch.relu(layer(marketStateBatch))
            transformGate = torch.sigmoid(gate(marketStateBatch))
            marketStateBatch = (carryGate * transformGate) + (marketStateBatch * (1 - transformGate))
        
        return self.final(marketStateBatch)


class TransformerBlock(nn.Module):
    def __init__(self, numLayers, numHeads, dimensionSize, attentionHeadOutputDimension, feedforward_dimensions):
        super().__init__()
        self.numLayers = numLayers
        self.encoderLayers = nn.ModuleList([
            Encoder(numHeads, dimensionSize, attentionHeadOutputDimension, feedforward_dimensions) for i in range(numLayers)
        ])

        # Feature attention to learn important features
        self.featureAttention = FeatureAttention(dimensionSize = 32, hiddenDimension = 16)

        # Input projection with residual
        self.inputProj = nn.Sequential(
            nn.Linear(dimensionSize, dimensionSize),
            nn.LayerNorm(dimensionSize),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.inputSkip = nn.Identity()

        # Advanced positional encoding
        self.positionEncoder = PositionalEncoding(dimensionSize = 32, timeLength = 100, dropout=0.1)

        self.cnnLayer = nn.Conv1d(dimensionSize, dimensionSize, kernel_size=3, padding=1)
        self.convNormalization = nn.LayerNorm(dimensionSize)

    def forward(self, marketStateBatch):
        #Projection normalization first
        inputProj = self.inputProj(marketStateBatch)
        inputSkip = self.inputSkip(marketStateBatch)
        inputTokens = inputProj + inputSkip

        #Adding feature attention after
        inputTokens = self.featureAttention(inputTokens)
        
        # Add positional encoding
        inputTokens = self.positionEncoder(inputTokens)
        
        # Temporal convolution
        inputtoCNN = inputTokens.unsqueeze(0).transpose(1, 2)  # Add batch dim and transpose for conv1d
        cnnOutput = torch.relu(self.cnnLayer(inputtoCNN))
        inputtoCNN =inputtoCNN + cnnOutput  # Residual
        inputtoCNN = inputtoCNN.transpose(1, 2).squeeze(0)  # Remove batch dim
        inputTokens = self.convNormalization(inputtoCNN)
        
        # Transformer layers
        for individualEncoderLayer in self.encoderLayers:
            inputTokens = individualEncoderLayer(inputTokens)
        
        return inputTokens