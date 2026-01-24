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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        self.currentTransformer.to(self.device)
        self.marketStateCompressor.to(self.device)

        self.l1LossFunction = nn.L1Loss()
        self.mseLossFunction = nn.MSELoss()
        self.huberLossFunction = nn.HuberLoss(delta = 0.3)

        allParameters = (
            list(self.currentTransformer.parameters()) + 
            list(self.marketStateCompressor.parameters())
        )

        self.optimizer = torch.optim.Adam(allParameters, lr=0.005) 
        self.validator = ScorerStepByStep(validationFileDirectory)


    def training(self, trainingFile, numEpochs, batches):
        trainingSequences = []
        unique_seqs = trainingFile['seq_ix'].unique()
        
        for seq_ix in unique_seqs:
            seq_data = trainingFile[trainingFile['seq_ix'] == seq_ix].sort_values('step_in_seq')
            marketStates = seq_data.iloc[:, 3:35].values  # Features columns
            targetValues = seq_data.iloc[:, 35:].values  # Target columns
            need_pred = seq_data['need_prediction'].values
            
            for i in range(len(marketStates)):
                if need_pred[i]:
                    contextWindow = marketStates[max(0, i-99):i+1]

                    if len(contextWindow) < 100:
                        padding = np.zeros((100 - len(contextWindow), contextWindow.shape[1]))
                        contextWindow = np.vstack([padding, contextWindow])
                    
                    trainingSequences.append({
                        'context': contextWindow,
                        'target': targetValues[i]
                    })

        
        bestvalPearson = -1.0

        for epoch in range(numEpochs):
            # Training phase
            self.currentTransformer.train()
            self.marketStateCompressor.train()
            
            # Shuffle and batch
            np.random.shuffle(trainingSequences)
            train_loss = 0.0
            num_batches = 0
            
            for batchStart in range(0, len(trainingSequences), batches):
                batchEnd = min(batchStart + batches, len(trainingSequences))
                batch = trainingSequences[batchStart:batchEnd]
                
                self.optimizer.zero_grad()
                batch_loss = 0.0
                for sample in batch:
                    contextWindow = torch.tensor(sample['context'], dtype=torch.float32).to(self.device)
                    targetValues = torch.tensor(sample['target'], dtype=torch.float32).to(self.device)
                    
                    transformerOutput = self.currentTransformer(contextWindow)
                    lastTimeStep = transformerOutput[-1, :]
                    prediction = self.marketStateCompressor(lastTimeStep)
                    prediction = torch.clamp(prediction, -6.0, 6.0)
                    
                    lossValue = ((self.l1LossFunction(prediction, targetValues) * 0.3) + 
                                 (self.mseLossFunction(prediction, targetValues) * 0.4) + 
                                 (self.huberLossFunction(prediction, targetValues) * 0.3)
                    )

                    batch_loss += lossValue
                
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.currentTransformer.parameters()) + 
                    list(self.marketStateCompressor.parameters()), 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                train_loss += batch_loss.item()
                num_batches += 1
            
            train_loss /= num_batches 

            valPearson = self.validator.score(self) 
            if valPearson > bestvalPearson:
                bestvalPearson = valPearson
                
                self.saveParameters('bestParams.pt', epoch, valPearson)
                print(f"Best model with Pearson: {valPearson:.6f}")

    def predict(self, currentSeq: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty(0, 32, dtype=torch.float32).to(self.device)
        
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
    
        prediction = finalPrediction.detach().cpu().numpy()
        return prediction

    def saveParameters(self, filename, epoch, valPearson):
        checkpoint = {
            'epoch': epoch,
            'modelState': self.currentTransformer.state_dict(),
            'compressorState': self.marketStateCompressor.state_dict(),
            'optimizerState': self.optimizer.state_dict(),
            'pearsonCoefficient': valPearson,
            'config': {
                'numLayers': 8,
                'numHeads': 8,
                'dimensionSize': 32,
                'feedforward_dimensions': 256,
            }
        }

        torch.save(checkpoint, filename)

    def loadParameters(self, filename):
        checkpoint = torch.load(filename, map_location = self.device)
        self.currentTransformer.load_state_dict(checkpoint['modelState'])
        self.marketStateCompressor.load_state_dict(checkpoint['compressorState'])

if __name__ == "__main__":
    if os.path.exists(validationFileDirectory):
        trialModel = PredictionModel()
        trialModel.training(trainingFile, 50, 32)

        if os.path.exists('bestParams.pt'):
            trialModel.loadParameters('bestParams.pt')

        scorer = ScorerStepByStep(validationFileDirectory)
        
        print("Testing Transformer...")
        results = scorer.score(trialModel)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet not found for testing.")