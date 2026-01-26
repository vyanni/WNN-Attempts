import os
import sys
from copy import deepcopy

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


def augment_sequence(sequence, scale_range=(0.95, 1.05), noise_std=0.01):
    """Data augmentation: random scaling and noise."""
    # Random scaling
    scale = np.random.uniform(scale_range[0], scale_range[1])
    aug_seq = sequence * scale
    
    # Random noise
    noise = np.random.normal(0, noise_std, sequence.shape)
    aug_seq = aug_seq + noise
    
    return aug_seq


class PredictionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_seq_ix = None
        self.sequence_history = []

        self.currentTransformer = transformer.TransformerBlock(
            numLayers=8, 
            numHeads=8, 
            dimensionSize=32, 
            attentionHeadOutputDimension=32, 
            feedforward_dimensions=256
        ).to(self.device)

        self.marketStateCompressor = transformer.HighwayNetwork(
            dimensionSize=32, 
            outputDimensions=2
        ).to(self.device)

        # Multi-component loss function
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=0.3)
        self.l1_loss = nn.L1Loss()

        # Combined optimizer
        all_params = list(self.currentTransformer.parameters()) + list(self.marketStateCompressor.parameters())
        self.optimizer = torch.optim.AdamW(all_params, lr=0.0001, weight_decay=0.02)
        
        # Learning rate scheduler
        self.scheduler = None
        
        # For SWA (Stochastic Weight Averaging)
        self.swa_transformer = None
        self.swa_compressor = None
        self.swa_count = 0

    def compute_combined_loss(self, predictions, targets):
        """Multi-component loss function."""
        mse = self.mse_loss(predictions, targets)
        huber = self.huber_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)
        
        # Weighted combination
        total_loss = 0.4 * mse + 0.3 * huber + 0.3 * l1
        return total_loss

    def train_model(self, num_epochs=50, batch_size=32, augmentation_factor=5):
        """Train the model with proper epoch-based training."""
        print(f"Training on device: {self.device}")
        
        # Prepare training data
        print("Preparing training data...")
        train_sequences = []
        unique_seqs = trainingFile['seq_ix'].unique()
        
        for seq_ix in unique_seqs:
            seq_data = trainingFile[trainingFile['seq_ix'] == seq_ix].sort_values('step_in_seq')
            states = seq_data.iloc[:, 3:35].values  # Features columns
            targets = seq_data.iloc[:, 35:].values  # Target columns
            need_pred = seq_data['need_prediction'].values
            
            for i in range(len(states)):
                if need_pred[i]:
                    # Use last 100 steps
                    start_idx = max(0, i - 99)
                    context = states[start_idx:i+1]
                    
                    # Pad to 100
                    if len(context) < 100:
                        padding = np.zeros((100 - len(context), context.shape[1]))
                        context = np.vstack([padding, context])
                    
                    train_sequences.append({
                        'context': context,
                        'target': targets[i]
                    })
        
        print(f"Total training samples: {len(train_sequences)}")
        
        # Data augmentation: create 5x more samples
        print(f"Augmenting data {augmentation_factor}x...")
        augmented_sequences = train_sequences.copy()
        for _ in range(augmentation_factor - 1):
            for sample in train_sequences:
                aug_context = augment_sequence(sample['context'].copy())
                augmented_sequences.append({
                    'context': aug_context,
                    'target': sample['target'].copy()
                })
        
        print(f"Total augmented samples: {len(augmented_sequences)}")
        
        # Initialize SWA models
        self.swa_transformer = deepcopy(self.currentTransformer)
        self.swa_compressor = deepcopy(self.marketStateCompressor)
        
        best_val_pearson = -1.0
        patience = 8
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.currentTransformer.train()
            self.marketStateCompressor.train()
            
            # Shuffle and batch
            np.random.shuffle(augmented_sequences)
            train_loss = 0.0
            num_batches = 0
            
            for batch_start in range(0, len(augmented_sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(augmented_sequences))
                batch = augmented_sequences[batch_start:batch_end]
                
                self.optimizer.zero_grad()
                
                batch_loss = 0.0
                for sample in batch:
                    context = torch.tensor(sample['context'], dtype=torch.float32).to(self.device)
                    target = torch.tensor(sample['target'], dtype=torch.float32).to(self.device)
                    
                    # Forward pass
                    transformer_out = self.currentTransformer(context)
                    last_timestep = transformer_out[-1, :]
                    prediction = self.marketStateCompressor(last_timestep)
                    prediction = torch.clamp(prediction, -6.0, 6.0)
                    
                    # Loss
                    loss = self.compute_combined_loss(prediction, target)
                    batch_loss += loss
                
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
            
            # Validation phase
            val_pearson = self._validate()
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Pearson: {val_pearson:.6f}")
            
            # SWA: Average weights from last 5 epochs
            if epoch >= (num_epochs - 5):
                self._update_swa_model()
            
            # Early stopping
            if val_pearson > best_val_pearson:
                best_val_pearson = val_pearson
                patience_counter = 0
                self.save_checkpoint('best_model.pt', epoch, val_pearson)
                print(f"  → Saved best model with Pearson: {val_pearson:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Apply SWA (use averaged weights)
        self._apply_swa()
        print("Applied SWA (Stochastic Weight Averaging)")
        
        # Save final SWA model
        self.save_checkpoint('swa_model.pt', num_epochs, best_val_pearson)
        print(f"Saved SWA model")

    def _update_swa_model(self):
        """Update SWA model with current model weights."""
        if self.swa_count == 0:
            self.swa_transformer = deepcopy(self.currentTransformer)
            self.swa_compressor = deepcopy(self.marketStateCompressor)
        else:
            # Running average for transformer
            for swa_param, param in zip(self.swa_transformer.parameters(), self.currentTransformer.parameters()):
                swa_param.data = (swa_param.data * self.swa_count + param.data) / (self.swa_count + 1)
            
            # Running average for compressor
            for swa_param, param in zip(self.swa_compressor.parameters(), self.marketStateCompressor.parameters()):
                swa_param.data = (swa_param.data * self.swa_count + param.data) / (self.swa_count + 1)
        
        self.swa_count += 1

    def _apply_swa(self):
        """Replace model weights with SWA averaged weights."""
        for swa_param, param in zip(self.swa_transformer.parameters(), self.currentTransformer.parameters()):
            param.data = swa_param.data
        
        for swa_param, param in zip(self.swa_compressor.parameters(), self.marketStateCompressor.parameters()):
            param.data = swa_param.data

    def _validate(self):
        """Validate on validation set."""
        self.currentTransformer.eval()
        self.marketStateCompressor.eval()
        
        all_predictions = []
        all_targets = []
        
        unique_seqs = validationFile['seq_ix'].unique()[:100]  # Use subset for speed
        
        with torch.no_grad():
            for seq_ix in unique_seqs:
                seq_data = validationFile[validationFile['seq_ix'] == seq_ix].sort_values('step_in_seq')
                states = seq_data.iloc[:, 3:35].values
                targets = seq_data.iloc[:, 35:].values
                need_pred = seq_data['need_prediction'].values
                
                for i in range(len(states)):
                    if need_pred[i]:
                        start_idx = max(0, i - 99)
                        context = states[start_idx:i+1]
                        
                        if len(context) < 100:
                            padding = np.zeros((100 - len(context), context.shape[1]))
                            context = np.vstack([padding, context])
                        
                        context = torch.tensor(context, dtype=torch.float32).to(self.device)
                        
                        transformer_out = self.currentTransformer(context)
                        last_timestep = transformer_out[-1, :]
                        prediction = self.marketStateCompressor(last_timestep)
                        prediction = torch.clamp(prediction, -6.0, 6.0).cpu().numpy()
                        
                        all_predictions.append(prediction)
                        all_targets.append(targets[i])
        
        if not all_predictions:
            return 0.0
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Compute weighted Pearson for both targets and average
        pearson_t0 = weighted_pearson_correlation(all_targets[:, 0], all_predictions[:, 0])
        pearson_t1 = weighted_pearson_correlation(all_targets[:, 1], all_predictions[:, 1])
        
        return (pearson_t0 + pearson_t1) / 2.0

    def save_checkpoint(self, filename, epoch, val_pearson):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.currentTransformer.state_dict(),
            'compressor_state_dict': self.marketStateCompressor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_pearson': val_pearson,
            'config': {
                'numLayers': 8,
                'numHeads': 8,
                'dimensionSize': 32,
                'feedforward_dimensions': 256,
            }
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """Load model from checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.currentTransformer.load_state_dict(checkpoint['model_state_dict'])
        self.marketStateCompressor.load_state_dict(checkpoint['compressor_state_dict'])
        print(f"Loaded checkpoint from {filename} (epoch {checkpoint['epoch']}, Pearson: {checkpoint['val_pearson']:.6f})")

    def training(self, currentSeq: DataPoint, targetValue):
        """Single sample training (for compatibility)."""
        if self.current_seq_ix != currentSeq.seq_ix:
            self.current_seq_ix = currentSeq.seq_ix
            self.sequence_history = torch.empty(0, 32, dtype=torch.float32, device=self.device)
        
        pytorchMarketState = torch.tensor(currentSeq.state.copy(), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.sequence_history = torch.cat([self.sequence_history, pytorchMarketState], dim=0)

        if not currentSeq.need_prediction:
            return None

        transformerOutput = self.currentTransformer(self.sequence_history[-100:, :])
        lastTimeStep = transformerOutput[-1, :]
        finalPrediction = self.marketStateCompressor(lastTimeStep)
        finalPrediction = torch.clamp(finalPrediction, -6.0, 6.0)
    
        prediction = finalPrediction.detach().cpu().numpy()

        targetValue = torch.tensor(targetValue, dtype=torch.float32, device=self.device)
        lossValue = self.compute_combined_loss(finalPrediction, targetValue)
        
        self.optimizer.zero_grad()
        lossValue.backward()

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
            self.sequence_history = torch.empty(0, 32, dtype=torch.float32, device=self.device)
        
        pytorchMarketState = torch.tensor(currentSeq.state.copy(), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.sequence_history = torch.cat([self.sequence_history, pytorchMarketState], dim=0)

        if not currentSeq.need_prediction:
            return None
        
        with torch.no_grad():
            transformerOutput = self.currentTransformer(self.sequence_history[-100:, :])
            lastTimeStep = transformerOutput[-1, :]
            finalPrediction = self.marketStateCompressor(lastTimeStep)
            finalPrediction = torch.clamp(finalPrediction, -6.0, 6.0)
        
        prediction = finalPrediction.cpu().numpy()    
        return prediction


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    model = PredictionModel()
    
    if args.train:
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        model.train_model(num_epochs=args.epochs, batch_size=args.batch_size, augmentation_factor=5)
    else:
        print("=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        
        # Try to load best model
        if os.path.exists('best_model.pt'):
            model.load_checkpoint('best_model.pt')
        
        if os.path.exists(validationFileDirectory):
            scorer = ScorerStepByStep(validationFileDirectory)
            print("Evaluating model...")
            results = scorer.score(model)
            
            print("\nResults:")
            print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
            for target in scorer.targets:
                print(f"  {target}: {results[target]:.6f}")
        else:
            print("Validation file not found.")
