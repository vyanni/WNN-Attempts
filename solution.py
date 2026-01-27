import os
import sys

#from example_solution.utils import DataPoint, ScorerStepByStep, weighted_pearson_correlation

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import pyarrow as pyarrow

from tqdm.auto import tqdm

ONLINE_NOTEBOOK = True

if ONLINE_NOTEBOOK:
    trainingFileDirectory = "/kaggle/input/lob-datasets/train.parquet"
    validationFileDirectory = "/kaggle/input/lob-datasets/valid.parquet"
    CHECKPOINT_DIR = "/kaggle/working/"
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(f"{CURRENT_DIR}/..")

    trainingFileDirectory = f"{CURRENT_DIR}\\datasets\\train.parquet"
    validationFileDirectory = f"{CURRENT_DIR}\\datasets\\valid.parquet"


trainingFile = pd.read_parquet(trainingFileDirectory)
validationFile = pd.read_parquet(validationFileDirectory)

class PredictionModel:
    def __init__(self):
        self.device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
        self.current_seq_ix = None
        self.sequence_history = []

        self.currentTransformer = TransformerBlock(
            numLayers = 8, 
            numHeads = 8, 
            dimensionSize = 32, 
            attentionHeadOutputDimension = 32, 
            feedforward_dimensions = 256
        )

        self.marketStateCompressor = HighwayNetwork(
            dimensionSize = 32, 
            outputDimensions = 2
        )
        #Brings it down from 1x32 to 1x2 

        self.currentTransformer.to(self.device)
        self.marketStateCompressor.to(self.device)

        self.l1LossFunction = nn.L1Loss()
        self.mseLossFunction = nn.MSELoss()
        self.huberLossFunction = nn.HuberLoss(delta = 0.3)

        self.allParameters = (
            list(self.currentTransformer.parameters()) + 
            list(self.marketStateCompressor.parameters())
        )

        self.optimizer = torch.optim.Adam(self.allParameters, lr=0.0001) 
        self.validator = ScorerStepByStep(validationFileDirectory)

        self.currentTransformer = torch.compile(
            self.currentTransformer,
            mode="max-autotune",
            dynamic = False
        )

        self.marketStateCompressor = torch.compile(
            self.marketStateCompressor,
            mode="max-autotune",
            dynamic = False
        )

    def batchGenerator(self, trainingFile, batchSize):
        contextWindows, targetValue = [], []
        unique_seqs = trainingFile['seq_ix'].unique()

        np.random.shuffle(unique_seqs)
        
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
                    
                    contextWindows.append(contextWindow)
                    targetValue.append(targetValues[i])
                
                if len(contextWindows) == batchSize:
                    yield contextWindows, targetValue
                    contextWindows, targetValue = [], []

        if contextWindows:
            yield contextWindows, targetValue
    
    def batchesCount(self, trainingFile, batchSize):
        count = 0
        for _ in self.batchGenerator(trainingFile, batchSize):
            count += 1
        return count

    def training(self, trainingFile, numEpochs, batchSize): 
        bestvalPearson = -1.0

        for epoch in tqdm(range(numEpochs), desc = "Epochs: ", position = 0, leave = True):
            # Training phase
            self.currentTransformer.train()
            self.marketStateCompressor.train()
            
            numBatches = 0
            totalBatches = self.batchesCount(trainingFile, batchSize)

            with tqdm(total = totalBatches, desc = "Total Batches: ") as pbar:
                for contexts_np, targets_np in self.batchGenerator(trainingFile, batchSize):

                    contextWindow = torch.tensor(
                        contexts_np, dtype=torch.float32, device=self.device
                    )

                    targetValues = torch.tensor(
                        targets_np, dtype=torch.float32, device=self.device
                    )


                    self.optimizer.zero_grad(set_to_none = True)
                    with torch.cuda.amp.autocast():
                        transformerOutput = self.currentTransformer(contextWindow)
                        lastTimeStep = transformerOutput[:, -1, :]

                        prediction = self.marketStateCompressor(lastTimeStep)
                        prediction = torch.clamp(prediction, -6.0, 6.0)
                    
                        lossValue = ((self.l1LossFunction(prediction, targetValues) * 0.3) + 
                                     (self.mseLossFunction(prediction, targetValues) * 0.4) + 
                                     (self.huberLossFunction(prediction, targetValues) * 0.3)
                        )
                
                    self.scaler.scale(lossValue).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.allParameters, 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    numBatches += 1
                    pbar.update(1)

                    if(numBatches % 150 == 0):                
                        valPearson = self.validator.score(self, True) 
                        print(f"Mean Weighted Pearson correlation for Batch {numBatches}: {valPearson['weighted_pearson']:.6f}")
                        for i, target in enumerate(self.validator.targets):
                            print(f"  {target}: {valPearson[target]:.6f}")
                    
                        weightedPearson = valPearson['weighted_pearson']
                        if weightedPearson > bestvalPearson:
                            bestvalPearson = weightedPearson

                            self.saveParameters('bestParams.pt', epoch, weightedPearson)
                            print(f"Best model with Pearson: {weightedPearson:.6f}")

            valPearson = self.validator.score(self, True) 
            weightedPearson = valPearson['weighted_pearson']
            if weightedPearson > bestvalPearson:
                bestvalPearson = weightedPearson

                self.saveParameters('bestParams.pt', epoch, weightedPearson)
                print(f"Best model with Pearson: {weightedPearson:.6f}")
    
    def predict(self, currentSeq: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
        
        pytorchMarketState = torch.tensor(currentSeq.state.copy(), dtype=torch.float32).unsqueeze(0).to(self.device)
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
            'device': self.device,
            'config': {
                'numLayers': 8,
                'numHeads': 8,
                'dimensionSize': 32,
                'feedforward_dimensions': 256,
            }
        }

        torch.save(checkpoint, filename)

    def loadParameters(self, filename):
        checkpoint = torch.load(filename, map_location = self.device, weights_only = False)
        self.currentTransformer.load_state_dict(checkpoint['modelState'])
        self.marketStateCompressor.load_state_dict(checkpoint['compressorState'])
        self.optimizer.load_state_dict(checkpoint['optimizerState'])
