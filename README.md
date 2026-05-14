# WNN-Attempts: Wunder Challenge - LOB Market Prediction

Transformer-based deep learning solution for predicting future price movements using Limit Order Book (LOB) data.

## Quick Start

```powershell
# 1. Activate environment (already set up)
.\trialEnv\Scripts\Activate.ps1

# 2. Install dependencies (first time only)
pip install torch numpy pandas pyarrow onnxruntime tqdm

# 3. Train model
python solution_v2.py

# 4. Convert to ONNX
python onnxMount.py

# 5. Evaluate
python solution.py
```

## Project Overview

**Challenge**: Predict two targets (t0, t1) representing future price movements based on 1000-step market sequences.

**Key Facts**:
- **Metric**: Weighted Pearson Correlation (emphasizes large price movements)
- **Dataset**: 10,721 training sequences + 1,444 validation sequences
- **Features**: 32 per timestep (bid/ask prices and volumes, trade data)
- **Targets**: 2 predictions per timestep
- **Input Window**: 100 timesteps (sequences start at step 99 after 99-step warmup)
- **Model**: 8-layer Transformer with feature attention and highway networks

## Data Structure

32 input features per timestep:
- **Bid/Ask Prices** (p0-p11): 12 features from LOB price levels
- **Bid/Ask Volumes** (v0-v11): 12 features from LOB volumes
- **Trade Data** (dp0-dp3, dv0-dv3): 8 features from recent trades

2 target outputs: t0 and t1 (future price movement indicators)

## 🏗️ Model Architecture

**Core Components**:
- **PositionalEncoding**: Sinusoidal + learnable embeddings with adaptive weighting
- **FeatureAttention**: Learns which of 32 features are important
- **8 Transformer Layers**: Multi-head attention (4 heads), layer norm, residuals
- **TemporalCNN**: 1D convolution for local pattern detection
- **HighwayNetwork**: 2-layer gated predictor with learnable skip connections
- **Multi-Loss**: 40% MSE + 30% Huber + 30% L1 for robust training
- **SWA**: Stochastic Weight Averaging after training for better generalization

See [transformer.py](transformer.py) for implementation details.

## Environment Setup

**Install Dependencies**:
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas pyarrow onnx onnxruntime onnx2torch tqdm
```

## Training

**Run Training**:
```bash
python solution_v2.py
```

**Training Features**:
- Multi-component loss function (MSE + Huber + L1)
- Data augmentation (5x expansion with random scaling + noise)
- Automatic GPU/CPU device selection
- Sequence-based train/validation split
- Model checkpointing (saves best to `bestParams.pt`)
- Expected time: 30-45 min on RTX 3090, 1-2 hours on V100

See [solution_v2.py](solution_v2.py) for training configuration and parameters.

## ONNX Models

**Convert to ONNX**:
```bash
python onnxMount.py
```

Creates `fullModel.onnx` (~50 MB) for fast inference (2-5x speedup compared to PyTorch).

**Advantages**: Framework-agnostic, optimized runtime, works on CPU/GPU/mobile, ~13 MB quantized.

## 📈 Evaluation & Metrics

**Evaluate on Validation Data**:
```bash
python solution.py
```

**Metric**: Weighted Pearson Correlation
- Weights predictions by absolute target value (emphasizes large price movements)
- Clipped to [-6, 6] to handle outliers
- Average of t0 and t1 scores
- Range: -1 to +1 (higher is better)

## Submission Format

**Required Structure**:
```
solution.zip
├── solution.py           # PredictionModel class with predict() method
├── fullModel.onnx        # Trained ONNX model
└── utils.py              # Import from example_solution/
```
## Project Structure

| Directory/File | Purpose |
|---|---|
| `transformer.py` | Model architecture (Transformer, attention, highway networks) |
| `solution_v2.py` | Training script with augmentation and multi-loss |
| `solution.py` | ONNX-based inference for evaluation/submission |
| `onnxMount.py` | Convert trained model to ONNX format |
| `bestParams.pt` | Trained model weights |
| `fullModel.onnx` | ONNX inference model |
| `example_solution/` | Reference baseline solution, utils, datasets |
| `QUICK_REFERENCE.md` | Cheat sheet for common tasks |

---

**Challenge**: Wunder Challenge 2 - Market Prediction 
**Updated**: May 14, 2026
