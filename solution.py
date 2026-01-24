import os
import sys

from example_solution.utils import DataPoint, ScorerStepByStep, weighted_pearson_correlation
import transformer

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

        self.currentTransformer = transformer.TransformerBlock(
            numLayers = 8, 
            numHeads = 8, 
            dimensionSize = 32, 
            attentionHeadOutputDimension = 32, 
            feedforward_dimensions = 256
        )

        self.marketStateCompressor = transformer.HighwayNetwork(
            dimensionSize = 32, 
            outputDimensions = 2
        )
        #Brings it down from 1x32 to 1x2 

        self.lossFunction = nn.L1Loss()

        allParameters = (
            list(self.currentTransformer.parameters()) + 
            list(self.marketStateCompressor.parameters())
        )

        self.optimizer = torch.optim.Adam(allParameters, lr=0.001)


    def training(self, currentSeq: DataPoint, targetValue):
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty(0, 32, dtype=torch.float32)
        
        pytorchMarketState = torch.tensor(currentSeq.state.copy(), dtype=torch.float32).unsqueeze(0)
        self.sequence_history = torch.cat([self.sequence_history, pytorchMarketState], dim = 0)

        if not currentSeq.need_prediction:
            return None

        self.currentTransformer.train()
        transformerOutput = self.currentTransformer(self.sequence_history[-100:, :])
        #Goes through the whole transformer process, with attention etc, outputs 100x32 matrix

        lastTimeStep = transformerOutput[-1, :]
        #Turns it into a 1x32 matrix for just a single timestep, use the last timestep for prediction

        self.marketStateCompressor.train()
        finalPrediction = self.marketStateCompressor(lastTimeStep)
        finalPrediction = torch.clamp(finalPrediction, -6.0, 6.0)
        #Changes it into a 1x2 vector for the predictions of t0 and t1
    
        prediction = finalPrediction.detach().numpy()

        targetValue = torch.tensor(targetValue, dtype=torch.float32)
        lossValue = self.lossFunction(finalPrediction, targetValue)
        
        self.optimizer.zero_grad()
        lossValue.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.currentTransformer.parameters()) + 
            list(self.marketStateCompressor.parameters()), 
            max_norm=1.0
        )
        
        self.optimizer.step()

        return prediction

    def predict(self, currentSeq: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty(0, 32, dtype=torch.float32)
        
        pytorchMarketState = torch.tensor(currentSeq.state.copy(), dtype=torch.float32).unsqueeze(0)
        self.sequence_history = torch.cat([self.sequence_history, pytorchMarketState], dim = 0)

        if not currentSeq.need_prediction:
            return None
        
        #Both are vector of size 100x32 for the 32 dimension market state, by the past 100 time steps
        #The transformer always takes in a 100x32 vector, where each row of 1x32 is 1 token
        #Its declared as (rows, columns), but mathematically, its columns x rows

        with torch.no_grad():
            self.currentTransformer.eval()
            transformerOutput = self.currentTransformer(self.sequence_history[-100:, :])
            #Goes through the whole transformer process, with attention etc, outputs 100x32 matrix

            lastTimeStep = transformerOutput[-1, :]
            #Turns it into a 1x32 matrix for just a single timestep

            self.marketStateCompressor.eval()
            finalPrediction = self.marketStateCompressor(lastTimeStep)
            finalPrediction = torch.clamp(finalPrediction, -6.0, 6.0)
            #Changes it into a 1x2 vector for the predictions of t0 and t1
    
        prediction = finalPrediction.detach().numpy()
        return prediction

if __name__ == "__main__":
    if os.path.exists(validationFileDirectory):
        trialModel = PredictionModel()

        amountSteps = 2000

        count = 0
        for idx, row in trainingFile.iterrows():
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            lob_data = row[3:35].to_numpy()
            labels = row[35:]

            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, lob_data)
            prediction = trialModel.training(data_point, labels)

            if(count % 50 == 0 and prediction is not None):
                errorPercentage = np.abs(prediction - labels) / labels
                print(errorPercentage)
            
            count += 1

            if(count >= amountSteps):
                print("Done Testing")
                break

        scorer = ScorerStepByStep(validationFileDirectory)
        
        print("Testing Transformer...")
        results = scorer.score(trialModel)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet not found for testing.")