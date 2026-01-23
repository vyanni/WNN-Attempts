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

trainingFileDirectory = f"{CURRENT_DIR}\\datasets\\train.parquet"
trainingFile = pd.read_parquet(trainingFileDirectory)

validationFileDirectory = f"{CURRENT_DIR}\\datasets\\valid.parquet"
validationFile = pd.read_parquet(validationFileDirectory)

class PredictionModel:
    def __init__(self):
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

    def training(self, currentSeq: DataPoint, targetValue):
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty(0, 32, dtype=torch.float32)
        
        pytorchMarketState = torch.tensor(currentSeq.state.copy(), dtype=torch.float32).unsqueeze(0)
        self.sequence_history = torch.cat([self.sequence_history, pytorchMarketState], dim = 0)

        if not currentSeq.need_prediction:
            return None

        inputTokens = self.sequence_history[-100:, :] + self.positionalEncodingVector
        
        transformerOutput = self.currentTransformer(inputTokens)
        #Goes through the whole transformer process, with attention etc, outputs 100x32 matrix

        singleTimeStep = torch.mean(transformerOutput, dim = 0)
        #Turns it into a 1x32 matrix for just a single timestep

        finalPrediction = self.finalLinear(singleTimeStep)
        #Changes it into a 1x2 vector for the predictions of t0 and t1
    
        prediction = finalPrediction.detach().numpy()
        errorPercentage = np.abs(prediction - targetValue) / targetValue
        print(errorPercentage)

        targetValue = torch.tensor(targetValue, dtype=torch.float32)

        self.lossValue = self.lossFunction(finalPrediction, targetValue)
        self.lossValue.backward()

        self.optimizerFunction.step()
        self.optimizerFunction.zero_grad()

        self.finalOptimizer.step()
        self.finalOptimizer.zero_grad()

    def predict(self, currentSeq: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty(0, 32, dtype=torch.float32)
        
        pytorchMarketState = torch.tensor(currentSeq.state.copy(), dtype=torch.float32).unsqueeze(0)
        self.sequence_history = torch.cat([self.sequence_history, pytorchMarketState], dim = 0)

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
    
        prediction = finalPrediction.detach().numpy()
        return prediction
    
trialModel = PredictionModel()

count = 0
for idx, row in trainingFile.iterrows():
    seq_ix = row[0]
    step_in_seq = row[1]
    need_prediction = row[2]
    lob_data = row[3:35].to_numpy()
    labels = row[35:]

    data_point = DataPoint(seq_ix, step_in_seq, need_prediction, lob_data)
    trialModel.training(data_point, labels)

    print(DataPoint)
    print(f"Iteration: {count}")
    count += 1

    if(count >= 10000):
        break

scorer = ScorerStepByStep(validationFileDirectory)
results = scorer.score(trialModel)