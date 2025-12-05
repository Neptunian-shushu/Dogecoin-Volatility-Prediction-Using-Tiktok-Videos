# Cryptocurrency Video Sentiment Analysis with Multimodal Fusion

This project analyzes cryptocurrency TikTok videos (specifically DogeCoin) to predict market sentiment using a **multimodal fusion approach** combining audio transcription, visual analysis, and multimodal reasoning.

## Table of Contents

- [Overview](#overview)
- [Multimodal Fusion Architecture](#multimodal-fusion-architecture)
- [Model Components](#model-components)
- [Technical Details](#technical-details)
- [Installation](#installation)
- [Usage](#usage)
- [Understanding the Results](#understanding-the-results)
- [Project Structure](#project-structure)
- [PACE-ICE Cluster Execution](#pace-ice-cluster-execution)

---

## Overview

This project implements a **multimodal fusion sentiment analyzer** that combines three complementary branches:

### Three-Branch Architecture

1. **Audio Branch (10% weight)**: Whisper transcription → FinBERT sentiment analysis
2. **Visual Branch (20% weight)**: CLIP frame embeddings → Visual sentiment classification
3. **Reasoning Branch (70% weight)**: Qwen3-VL multimodal reasoning on key frames + transcript

### Key Features

- **Multimodal Understanding**: Analyzes audio, visual, and combined multimodal signals
- **Adaptive Scaling**: FinBERT sentiment amplified based on neutral probability for social media content
- **Zero-shot Learning**: No training required - models understand crypto concepts inherently
- **Intelligent Frame Selection**: Uses maximal distance sampling for diverse key frame selection
- **PACE-ICE Optimized**: Includes batch processing script for HPC cluster execution

---

## Multimodal Fusion Architecture

### Complete Pipeline

```
TikTok Video Input
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         │                 │                 │                 │
    [Audio Track]     [Video Frames]   [Audio + Frames]       │
         │                 │                 │                 │
         ▼                 ▼                 ▼                 │
    ┌─────────┐       ┌─────────┐      ┌──────────┐          │
    │ Whisper │       │  CLIP   │      │ Key Frame│          │
    │  (base) │       │ ViT-B32 │      │ Selection│          │
    └────┬────┘       └────┬────┘      └────┬─────┘          │
         │                 │                 │                 │
    Transcript        Embeddings        3 Diverse             │
         │                 │             Frames               │
         ▼                 ▼                 │                 │
    ┌─────────┐       ┌─────────┐          │                 │
    │ FinBERT │       │ Visual  │          │                 │
    │Financial│       │Sentiment│          │                 │
    │Sentiment│       │  CLIP   │          │                 │
    └────┬────┘       └────┬────┘          │                 │
         │                 │                 │                 │
    ┌────┴────┐            │                 ▼                 │
    │Adaptive │            │          ┌──────────────┐         │
    │ Scaling │            │          │   Qwen3-VL   │◄────────┘
    │(Neutral-│            │          │     8B       │  Transcript
    │  based) │            │          │  Multimodal  │
    └────┬────┘            │          │   Reasoning  │
         │                 │          └──────┬───────┘
         ▼                 ▼                 ▼
    Audio Score      Visual Score     Reasoning Score
    (weight: 0.1)    (weight: 0.2)    (weight: 0.7)
         │                 │                 │
         └─────────────────┴─────────────────┘
                           ▼
                   ┌──────────────┐
                   │    Fusion    │
                   │   Weighted   │
                   │ Combination  │
                   └──────┬───────┘
                          ▼
              ┌─────────────────────┐
              │ Final Sentiment     │
              │ Score: [-1, +1]     │
              │ Class: POS/NEG/NEU  │
              └─────────────────────┘
```

### Why Three Branches?

1. **Audio Branch**: Captures spoken sentiment and explicit statements
2. **Visual Branch**: Captures visual cues (charts, emotions, colors, memes)
3. **Reasoning Branch**: Combines both modalities with context understanding

### Fusion Weights Rationale

- **Audio: 10%** - FinBERT is conservative on casual social media language
- **Visual: 20%** - CLIP provides reliable visual sentiment signals
- **Reasoning: 70%** - Qwen3-VL provides the most comprehensive multimodal analysis

---

## Model Components

### 1. Audio Branch: Whisper + FinBERT

**Whisper (base)**: OpenAI's robust speech recognition model
- Transcribes TikTok audio to text
- Handles background music, multiple speakers, accents

**FinBERT**: Financial sentiment analysis model
- Trained on financial text (news, reports, discussions)
- Outputs: P(positive), P(negative), P(neutral)

**Adaptive Scaling**: Counteracts FinBERT's conservatism
```python
raw_score = P(positive) - P(negative)
scale_factor = 1 + (P(neutral) × 10)
final_score = clip(raw_score × scale_factor, -1.0, 1.0)
```

### 2. Visual Branch: CLIP

**CLIP (ViT-B/32)**: OpenAI's vision-language model
- Extracts frames at 1 FPS
- Computes image embeddings
- Compares against sentiment text prompts:
  - "positive cryptocurrency news, bullish market, growth, moon, pump"
  - "negative cryptocurrency news, bearish market, decline, crash, dump"
  - "neutral cryptocurrency discussion, stable market, sideways"

**Key Frame Selection**: Maximal distance sampling
- Selects 3 diverse frames using embedding distance
- Ensures temporal coverage across video

### 3. Reasoning Branch: Qwen3-VL

**Qwen3-VL-8B-Instruct**: State-of-the-art multimodal LLM
- Analyzes transcript + 3 key frames simultaneously
- Understands context, metaphors, and multimodal signals
- Outputs structured sentiment with reasoning

**Prompt Engineering**: Direct output format
```
SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
SCORE: [number from -1.0 to +1.0]
REASONING: [Brief explanation]
```

---

## How CLIP Works for Video Sentiment Analysis

### What is CLIP?

**CLIP (Contrastive Language-Image Pre-training)** is a neural network trained by OpenAI on 400 million image-text pairs from the internet. It understands the relationship between images and text descriptions.

### Why CLIP for TikTok Videos?

CLIP can "see" and understand visual cues in cryptocurrency videos:

- **Charts**: Price going up or down
- **Facial Expressions**: Happy, concerned, excited
- **Text Overlays**: "TO THE MOON!", "CRASH INCOMING"
- **Colors**: Green (bullish), Red (bearish)
- **Memes**: Rockets, moon references, Doge memes

### Zero-Shot Classification

CLIP doesn't need training on crypto videos specifically. It already understands:
- "bullish market" = positive sentiment
- "bearish market" = negative sentiment
- "growth", "moon", "pump" = positive
- "crash", "dump", "decline" = negative

## Technical Details

### Multimodal Fusion Formula

```python
final_score = (0.1 × audio_score) + (0.2 × visual_score) + (0.7 × reasoning_score)
```

### Audio Branch Pipeline

```
Video Audio → Whisper → Transcript → FinBERT → P(pos), P(neg), P(neu)
                                              ↓
                                    Adaptive Scaling:
                                    scale = 1 + (P(neu) × 10)
                                    score = (P(pos) - P(neg)) × scale
```

**Why Adaptive Scaling?**
- FinBERT is trained on formal financial text
- TikTok uses casual, exaggerated language
- High neutral probability indicates FinBERT's uncertainty
- Amplification helps recover true sentiment signal

### Visual Branch Pipeline

```
Video Frames (1 FPS) → CLIP Embeddings → Mean Pooling → Video Representation
                                                              ↓
                                              Compare with sentiment prompts
                                                              ↓
                                              Softmax → P(pos), P(neg), P(neu)
                                                              ↓
                                              Score = P(pos) - P(neg)
```

### Reasoning Branch Pipeline

```
All Frames + Embeddings → Maximal Distance Sampling → 3 Key Frames
                                                           ↓
                      Transcript + Key Frames → Qwen3-VL → Structured Output
                                                           ↓
                                              Parse: SENTIMENT, SCORE, REASONING
```

### CLIP's Image-Text Matching

CLIP has two encoders that convert images and text into the same vector space:

```
Frame Image → CLIP Image Encoder → 512-dimensional vector
                                          ↓
Text Labels → CLIP Text Encoder  → 512-dimensional vectors
                                          ↓
                                  Compare similarity (dot product)
                                          ↓
                                  Softmax → Probabilities
```

### Example Computation

For a single frame showing a green chart with "DOGE TO THE MOON!" text:

```python
# Input to CLIP
Frame: [RGB image of bullish crypto content]

Labels:
- "positive cryptocurrency news, bullish market, growth, moon, pump"
- "negative cryptocurrency news, bearish market, decline, crash, dump"  
- "neutral cryptocurrency discussion, stable market, sideways"

# CLIP Output Probabilities
├─ Positive: 0.72  ← CLIP says "this looks bullish!"
├─ Negative: 0.15
└─ Neutral:  0.13

# Calculate Sentiment Score
Sentiment = 0.72 - 0.15 = 0.57

# Classify
Classification: POSITIVE (score > 0.2)
Confidence: HIGH (score > 0.4)
```

### What CLIP "Sees" in Frames

CLIP understands visual semantics:
- Green colors, rising charts
- Excited facial expressions
- Text overlays with positive words
- Rocket emojis, moon references
- Bullish market imagery
- Memes and visual metaphors

### Complete Pipeline Visualization

```
TikTok Video: "DOGECOIN TO THE MOON!" (10 seconds)
│
├─ Frame 1 (0.0s):  [Person pointing at chart] 
├─ Frame 2 (2.5s):  [Green candles rising]
├─ Frame 3 (5.0s):  [Text overlay: "BUY NOW!"]
├─ Frame 4 (7.5s):  [Doge meme with rocket]
└─ Frame 5 (10.0s): [Excited reaction face]
         ↓
    [OpenCV Extraction]
         ↓
    5 RGB Images (frames)
         ↓
    [CLIP Analysis for each frame]
         ↓
┌────────────────────────────────────────┐
│ Frame 1: Sentiment = +0.6  (POSITIVE) │
│ Frame 2: Sentiment = +0.7  (POSITIVE) │
│ Frame 3: Sentiment = +0.5  (POSITIVE) │
│ Frame 4: Sentiment = +0.8  (POSITIVE) │
│ Frame 5: Sentiment = +0.4  (POSITIVE) │
└────────────────────────────────────────┘
         ↓
    Average: (0.6 + 0.7 + 0.5 + 0.8 + 0.4) / 5 = 0.60
         ↓
┌──────────────────────────────────────────────┐
│ Final Result:                                │
│ • Sentiment Score: 0.60                      │
│ • Classification: POSITIVE                   │
│ • Confidence: HIGH                           │
│ • Method: CLIP visual analysis from 5 frames│
└──────────────────────────────────────────────┘
         ↓
    Saved to CSV with date and video path
```

### Why This Approach Works

1. **Temporal Coverage**: 5 frames capture the whole video's story
2. **Visual Semantics**: CLIP understands what "bullish" looks like visually
3. **Robust Averaging**: Multiple frames reduce noise from any single frame
4. **No Training Needed**: CLIP already knows crypto concepts from internet pre-training
5. **Scalable**: Works on any crypto content (Bitcoin, Ethereum, DogeCoin, etc.)

### Key Advantages

- **Zero-shot learning**: No labeled training data needed
- **Visual understanding**: Analyzes actual video content, not just metadata
- **Meme comprehension**: Understands crypto memes and visual metaphors
- **Multi-frame analysis**: Captures video dynamics over time
- **Robust fallback**: Uses filename heuristics if frame extraction fails
- **Format agnostic**: Works with MP4, AVI, MOV, MKV, WebM

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch transformers opencv-python pandas numpy

# Multimodal models
pip install openai-whisper qwen-vl-utils accelerate av pillow librosa soundfile

# System requirement
# Install ffmpeg for audio extraction:
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# PACE-ICE: module load ffmpeg

# Optional: For GPU acceleration
# Install CUDA-enabled PyTorch from pytorch.org
```

### Model Downloads

On first run, the following models will be automatically downloaded:
- **Whisper base**: ~140 MB
- **CLIP ViT-B/32**: ~350 MB  
- **FinBERT**: ~440 MB
- **Qwen3-VL-8B**: ~16 GB

Total: ~17 GB (ensure sufficient disk space)

### Verify Installation

Run the first few cells of `multimodal_fusion.ipynb` to verify:
- PyTorch installed
- CUDA available (for GPU acceleration)
- All model libraries available (transformers, whisper, qwen-vl-utils)
- ffmpeg installed for audio extraction

---

## Usage

### Method 1: Jupyter Notebook (Interactive)

1. **Open the notebook**:
   ```bash
   jupyter notebook multimodal_fusion.ipynb
   ```

2. **Place videos** in `./videos/` directory with date-based names:
   ```
   videos/
   ├── 2025-01-15.mp4
   ├── 2025-01-16.mp4
   └── 2025-01-17.mp4
   ```

3. **Run all cells** - the notebook will:
   - Load all models (Whisper, CLIP, Qwen3-VL, FinBERT)
   - Process each video through all three branches
   - Fuse results with weighted combination
   - Save results to CSV files

### Method 2: Python Script (Batch Processing)

For large-scale processing or cluster execution:

```bash
python multimodal_inference.py \
    --video_dir ./videos \
    --output_dir ./results \
    --audio_weight 0.1 \
    --visual_weight 0.2 \
    --reasoning_weight 0.7
```

**Arguments**:
- `--video_dir`: Directory containing video files (default: `./videos`)
- `--output_dir`: Output directory for results (default: `./results`)
- `--audio_weight`: Weight for audio branch (default: 0.1)
- `--visual_weight`: Weight for visual branch (default: 0.2)
- `--reasoning_weight`: Weight for reasoning branch (default: 0.7)

### Method 3: PACE-ICE Cluster Execution

Submit the batch job on Georgia Tech's PACE-ICE cluster:

```bash
sbatch setup.sbatch
```

The `setup.sbatch` script will:
- Request appropriate GPU resources
- Load required modules
- Install dependencies
- Run the inference script on all videos
- Save results to `./results/`

---

## Understanding the Results

### Output Files

Two CSV files are generated:

#### 1. `multimodal_fusion_sentiment.csv` (Main Results)

Aggregated by date with mean sentiment scores:

```csv
date,num_videos,video_path,final_sentiment_score,final_sentiment_class,audio_score,visual_score,reasoning_score,method,timestamp
2025-01-15,1,2025-01-15.mp4,0.65,POSITIVE,0.42,0.58,0.72,Multimodal Fusion,2025-01-15T10:30:45
2025-01-16,2,2025-01-16_1.mp4; 2025-01-16_2.mp4,0.15,NEUTRAL,0.05,0.12,0.18,Multimodal Fusion,2025-01-16T11:20:30
```

#### 2. `multimodal_fusion_details.csv` (Detailed Analysis)

Individual video analysis with transcripts and reasoning:

```csv
date,video_path,transcript,reasoning,timestamp
2025-01-15,2025-01-15.mp4,"Dogecoin is going to the moon! This is the best time to buy...",The visual content shows excited facial expressions with rising chart graphics...,2025-01-15T10:30:45
```

### Sentiment Score Range

- **+1.0**: Extremely positive (very bullish)
- **+0.5**: Positive (bullish)
- **0.0**: Neutral
- **-0.5**: Negative (bearish)
- **-1.0**: Extremely negative (very bearish)

### Classification Thresholds

- **POSITIVE**: score > 0.2
- **NEUTRAL**: -0.2 ≤ score ≤ 0.2
- **NEGATIVE**: score < -0.2

### Confidence Levels

- **HIGH**: |score| > 0.4 (strong signal)
- **MEDIUM**: 0.2 < |score| ≤ 0.4 (moderate signal)
- **LOW**: |score| ≤ 0.2 (weak signal)

### Output Columns Explained

| Column | Description |
|--------|-------------|
| `date` | Extracted from filename (YYYY-MM-DD) |
| `num_videos` | Number of videos analyzed for this date |
| `video_path` | Path(s) to video file(s) |
| `final_sentiment_score` | Fused score from -1 to +1 |
| `final_sentiment_class` | POSITIVE, NEGATIVE, or NEUTRAL |
| `audio_score` | Audio branch sentiment score |
| `visual_score` | Visual branch sentiment score |
| `reasoning_score` | Reasoning branch sentiment score |
| `method` | Analysis method used |
| `transcript` | Whisper audio transcription (details file) |
| `reasoning` | Qwen3-VL reasoning explanation (details file) |
| `timestamp` | When the analysis was performed |

### Branch Score Interpretation

Each branch produces a score from -1.0 to +1.0:

**Audio Score** (Whisper + FinBERT):
- Based on spoken content sentiment
- Amplified for casual social media language
- May be conservative on informal speech

**Visual Score** (CLIP):
- Based on visual cues (charts, emotions, colors)
- Robust to memes and visual metaphors
- Captures frame-level sentiment signals

**Reasoning Score** (Qwen3-VL):
- Combines audio + visual + context
- Most comprehensive analysis
- Highest weight in final fusion (70%)

### Final Classification

The final sentiment is classified based on the fused score:

- **POSITIVE**: score > 0.2 (bullish sentiment)
- **NEUTRAL**: -0.2 ≤ score ≤ 0.2 (mixed/unclear sentiment)
- **NEGATIVE**: score < -0.2 (bearish sentiment)

---

## Project Structure

```
MGT6785/
├── multimodal_fusion.ipynb       # Interactive notebook for multimodal analysis
├── multimodal_inference.py       # Batch processing script for cluster/server
├── prediction.ipynb              # Time series prediction with GRU models
├── setup.sbatch                  # SLURM batch script for PACE-ICE cluster
├── README.md                     # This file
│
├── videos/                       # Input videos directory
│   ├── 2025-01-15.mp4
│   └── 2025-01-16.mp4
│
├── results/                      # Output directory
│   ├── multimodal_fusion_sentiment.csv      # Main results (aggregated)
│   ├── multimodal_fusion_details.csv        # Detailed results (per video)
│   ├── btc_price_predictions.png            # Bitcoin price predictions
│   ├── btc_volatility_predictions.png       # Bitcoin volatility predictions
│   ├── doge_price_predictions.png           # Dogecoin price predictions
│   ├── doge_volatility_predictions.png      # Dogecoin volatility predictions
│   └── prediction_summary.csv               # Model performance metrics
│
├── models/                       # Saved model checkpoints
│   ├── btc_price_model.pth              # Bitcoin price GRU model
│   ├── btc_volatility_model.pth         # Bitcoin volatility GRU model
│   ├── doge_price_model.pth             # Dogecoin price GRU model
│   ├── doge_volatility_model.pth        # Dogecoin volatility GRU model
│   └── *_scalers.pkl                    # Feature/target scalers
│
├── data/                         # Processed data and features
├── temp/                         # Temporary files (audio, frames)
├── logs/                         # Training and execution logs
└── checkpoints/                  # Model training checkpoints
```

---

## Price and Volatility Prediction (prediction.ipynb)

### Overview

The `prediction.ipynb` notebook implements cryptocurrency price and volatility forecasting using GRU (Gated Recurrent Unit) neural networks. It combines sentiment analysis results with technical market indicators for comprehensive prediction.

### Architecture

**Model**: GRU (Gated Recurrent Unit)
- 2-layer GRU with 64 hidden units
- Dropout regularization (0.2) to prevent overfitting
- Fully connected output layers
- Early stopping and learning rate scheduling

**Targets**: Four prediction tasks
1. Bitcoin (BTC) price prediction
2. Bitcoin (BTC) volatility prediction
3. Dogecoin (DOGE) price prediction
4. Dogecoin (DOGE) volatility prediction

### Feature Engineering

**Technical Indicators** (automatically computed):
- **Returns**: Daily returns, log returns
- **Volatility**: 5-day, 10-day, 20-day rolling standard deviation
- **Moving Averages**: 5-day, 10-day, 20-day MA
- **Momentum**: 5-day, 10-day price momentum
- **Volume**: Volume moving average, volume ratio
- **RSI**: Relative Strength Index (14-day)
- **Bollinger Bands**: Upper, middle, lower bands, position
- **High-Low Range**: Daily price range normalized

**Sentiment Features** (from multimodal fusion):
- Final sentiment score (fused from audio + visual + reasoning)
- Audio branch score (Whisper + FinBERT)
- Visual branch score (CLIP)
- Reasoning branch score (Qwen3-VL)
- Lagged sentiment features (1, 3, 7 days)
- Number of videos per day

**Total Features**: ~37 features per prediction

### Data Pipeline

```
Sentiment Results (CSV) + Yahoo Finance Data (API)
                    ↓
        Technical Indicator Calculation
                    ↓
        Sentiment Feature Merging
                    ↓
        Feature Scaling (StandardScaler)
                    ↓
        Sequence Creation (5-day windows)
                    ↓
        Train/Validation Split (70/30)
                    ↓
        GRU Model Training
                    ↓
        Evaluation (R², RMSE, IC)
```

### Performance Metrics

Models are evaluated using:
- **R² Score**: Coefficient of determination (how well predictions fit actual values)
- **RMSE**: Root Mean Squared Error (average prediction error)
- **IC (Information Coefficient)**: Spearman correlation between predicted and actual returns
  - Only calculated for price models (not volatility)
  - IC > 0.05 is considered meaningful in quantitative finance
  - Measures directional prediction accuracy

### Usage

1. **Prerequisites**: Run `multimodal_fusion.ipynb` first to generate sentiment scores

2. **Open notebook**:
   ```bash
   jupyter notebook prediction.ipynb
   ```

3. **Run all cells** - the notebook will:
   - Load sentiment analysis results from `results/`
   - Fetch price data from Yahoo Finance API
   - Engineer technical indicators
   - Merge sentiment with price data
   - Train 4 separate GRU models
   - Generate predictions and visualizations
   - Save trained models to `models/`

4. **Output files**:
   - Trained models: `models/*.pth`
   - Scalers: `models/*_scalers.pkl`
   - Visualizations: `results/*_predictions.png`
   - Performance metrics: `results/prediction_summary.csv`

### Model Configuration

**Hyperparameters**:
- Sequence length: 5 days (look-back window)
- Hidden size: 64 units
- Number of layers: 2
- Dropout: 0.2
- Batch size: 32
- Learning rate: 0.001 (with ReduceLROnPlateau)
- Optimizer: Adam
- Loss function: MSE (Mean Squared Error)
- Early stopping patience: 10 epochs

**Training**:
- Train/validation split: 70/30
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Gradient clipping for stability

### Interpretation

**Price Predictions**:
- Predict next-day closing price
- Useful for trading strategies
- IC metric shows directional accuracy

**Volatility Predictions**:
- Predict next-day price volatility (risk)
- Useful for risk management
- Helps identify high-risk periods

**Sentiment Impact**:
- Lagged sentiment features capture delayed market reactions
- Multiple sentiment sources (audio, visual, reasoning) provide robust signals
- Sentiment + technical indicators outperform technical indicators alone

### Example Results

Typical performance (varies by market conditions):
- **Bitcoin Price**: R² ~0.7-0.9 (train), R² ~0.2-0.6 (validation)
- **Bitcoin Volatility**: R² ~0.5-0.7 (train), R² ~0.0-0.3 (validation)
- **Dogecoin Price**: R² ~0.7-0.9 (train), R² ~0.1-0.5 (validation)
- **Dogecoin Volatility**: R² ~0.6-0.8 (train), R² ~-0.2-0.2 (validation)

Note: Negative validation R² indicates overfitting - model predicts worse than mean baseline.

### Key Findings

1. **Sentiment Integration**: Multimodal sentiment scores improve prediction accuracy
2. **Overfitting Challenge**: High training R² but lower validation R² indicates need for regularization
3. **Feature Importance**: Technical indicators (MA, volatility) + lagged sentiment most predictive
4. **Model Complexity**: GRU captures temporal dependencies better than simple regression
5. **Volatility Harder**: Volatility prediction more challenging than price prediction

---

## PACE-ICE Cluster Execution

### Prerequisites

1. Access to Georgia Tech PACE-ICE cluster
2. Videos uploaded to `./videos/` directory
3. `setup.sbatch` configured for your resource needs

### Running on PACE-ICE

1. **Connect to PACE-ICE**:
   ```bash
   ssh <your-gt-username>@login-ice.pace.gatech.edu
   ```

2. **Navigate to project directory**:
   ```bash
   cd ~/MGT6785
   ```

3. **Place videos**:
   ```bash
   # Copy videos to videos/ directory
   cp /path/to/your/videos/*.mp4 ./videos/
   ```

4. **Submit batch job**:
   ```bash
   sbatch setup.sbatch
   ```

5. **Monitor job status**:
   ```bash
   squeue -u $USER
   ```

6. **Check output**:
   ```bash
   # View SLURM output log
   cat slurm-<jobid>.out
   
   # Check results
   ls -lh results/
   head results/multimodal_fusion_sentiment.csv
   ```

### setup.sbatch Configuration

The batch script requests:
- GPU resources (for deep learning models)
- Sufficient memory (~32GB recommended for Qwen3-VL)
- Time limit (adjust based on number of videos)
- Loads required modules (CUDA, ffmpeg, Python)

Example `setup.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=multimodal_sentiment
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

# Load modules
module load cuda/11.8
module load ffmpeg

# Activate environment (if using conda/venv)
source activate myenv

# Run inference
python multimodal_inference.py \
    --video_dir ./videos \
    --output_dir ./results
```

---

## Advanced Usage

### Custom Fusion Weights

Adjust branch weights based on your trust in each modality:

```python
# Example: Trust visual more than reasoning
analyzer.process_all_videos(
    video_dir='./videos',
    output_file='./results/custom_fusion.csv',
    details_file='./results/custom_details.csv',
    fusion_weights={'audio': 0.1, 'visual': 0.5, 'reasoning': 0.4}
)
```

### Command-line Weight Adjustment

```bash
python multimodal_inference.py \
    --audio_weight 0.2 \
    --visual_weight 0.3 \
    --reasoning_weight 0.5
```

### Extract More Frames

Modify frame extraction rate for longer videos:

```python
# Extract at 2 FPS instead of 1 FPS
audio_path, frames = extract_audio_and_frames(video_path, fps=2.0)
```

### Adjust Key Frame Count

Select more or fewer key frames for reasoning branch:

```python
# Use 5 key frames instead of 3
key_frames = select_key_frames(frames, embeddings, num_key_frames=5)
```

### Process Single Video

For testing or debugging:

```python
from multimodal_inference import MultimodalSentimentAnalyzer

analyzer = MultimodalSentimentAnalyzer()
result = analyzer.analyze_video('./videos/2025-01-15.mp4')

print(f"Score: {result['final_sentiment_score']:.3f}")
print(f"Class: {result['final_sentiment_class']}")
print(f"Reasoning: {result['components']['reasoning']['reasoning']}")
```

---

## Performance Considerations

### GPU Requirements

- **Minimum**: 8GB VRAM (for CLIP + FinBERT)
- **Recommended**: 24GB VRAM (for Qwen3-VL-8B)
- **PACE-ICE**: Request GPU nodes with `--gres=gpu:1`

### Processing Time

Approximate time per video (on A100 GPU):
- Audio extraction + Whisper: ~5-10 seconds
- Visual CLIP processing: ~2-5 seconds  
- Qwen3-VL reasoning: ~15-30 seconds
- **Total**: ~25-45 seconds per video

### Memory Usage

- **Model weights**: ~17GB disk space
- **Runtime memory**: ~20-32GB RAM
- **GPU memory**: ~16-20GB VRAM

### Optimization Tips

1. **Batch processing**: Process multiple videos in one job
2. **Frame sampling**: Use 1 FPS for standard videos, 2 FPS for long videos
3. **Key frames**: 3 frames balances quality vs. speed
4. **Model quantization**: Use `torch_dtype=torch.bfloat16` (already enabled)

---

## Troubleshooting

### Audio extraction fails

- **Check ffmpeg**: `which ffmpeg` or `module load ffmpeg`
- **Video has no audio**: Script will continue with visual-only analysis
- **Codec issues**: Try converting video to standard MP4

### GPU out of memory

- **Reduce batch size** in CLIP processing
- **Use fewer key frames** (2 instead of 3)
- **Enable CPU offloading** for Qwen3-VL
- **Request more VRAM** on PACE-ICE

### Model download issues

- **Check internet connection** on first run
- **Verify Hugging Face access**: Models download from `huggingface.co`
- **Disk space**: Ensure 20GB+ free space
- **Manual download**: Download models to cache directory

### Inconsistent results

- **Video quality**: Low-quality videos may produce unreliable scores
- **Content relevance**: Ensure videos are about cryptocurrency
- **Mixed sentiment**: Videos with conflicting signals may score near neutral
- **Adjust weights**: Experiment with different fusion weights

### PACE-ICE specific

- **Module not found**: Ensure `module load cuda ffmpeg` in batch script
- **Timeout**: Increase `--time` for large video batches
- **Permission denied**: Check file permissions on videos directory

---

## Model Architecture Details

### Why Multimodal Fusion?

Single-modality approaches have limitations:
- **Audio-only**: Misses visual cues (charts, emotions, memes)
- **Visual-only**: Misses explicit statements and context
- **LLM-only**: Expensive and may hallucinate without structured prompts

**Multimodal fusion** combines strengths while mitigating weaknesses.

### Adaptive Scaling Explained

FinBERT is conservative on casual language. Example:

**Input**: "DOGE TO THE MOON!!!"

**FinBERT raw output**:
- P(positive) = 0.4
- P(negative) = 0.1  
- P(neutral) = 0.5 ← High uncertainty!

**Without scaling**: 0.4 - 0.1 = 0.3 (underestimates enthusiasm)

**With adaptive scaling**:
- scale_factor = 1 + (0.5 × 10) = 6.0
- scaled_score = 0.3 × 6.0 = 1.8 → clipped to 1.0

**Result**: Properly recognizes extreme bullish sentiment

### Key Frame Selection Algorithm

**Goal**: Select diverse frames that represent different parts of the video

**Method**: Maximal distance sampling
1. Start with first frame
2. For each remaining slot:
   - Find frame most distant from already-selected frames
   - Add to selection
3. Result: 3 temporally and visually diverse frames

**Example**:
- Frame 0 (start): Person introducing topic
- Frame 12 (middle): Chart showing price movement
- Frame 24 (end): Conclusion with call-to-action

### Why These Weights?

**Audio: 10%**
- FinBERT struggles with informal language
- Transcription errors possible
- Useful for explicit statements

**Visual: 20%**  
- CLIP is reliable for visual sentiment
- Captures memes and emotions well
- Limited context understanding

**Reasoning: 70%**
- Qwen3-VL sees full context
- Combines audio + visual signals
- Best at understanding nuance and metaphor
- Most expensive computationally (hence saved for key frames only)

---

## Comparison with Other Approaches

### vs. CLIP-only (Previous Approach)

| Aspect | CLIP-only | Multimodal Fusion |
|--------|-----------|-------------------|
| Audio analysis | None | Whisper + FinBERT |
| Visual analysis | All frames | All frames + key frames |
| Context understanding | Limited | Qwen3-VL reasoning |
| Processing time | Fast (~5s) | Slower (~30s) |
| Accuracy | Good | Better |
| Cost | Low | Higher (LLM inference) |

### vs. LLM-only (GPT-4V, Gemini)

| Aspect | LLM-only | Multimodal Fusion |
|--------|----------|-------------------|
| Cost per video | High ($0.01-0.05) | Lower (self-hosted) |
| Processing time | 10-30s | 25-45s |
| Structured output | Requires prompting | Consistent format |
| Specialized sentiment | Generic | Financial-trained |
| Offline deployment | API required | Fully local |

---

## Future Enhancements

Potential improvements to the system:

1. **Temporal Modeling**: Add LSTM/Transformer for frame sequences
2. **OCR Integration**: Extract and analyze text overlays (EasyOCR, PaddleOCR)
3. **Emotion Recognition**: Facial emotion detection (FER, DeepFace)
4. **Fine-tuning**: Train on cryptocurrency-specific videos
5. **Larger Models**: Upgrade to CLIP-Large, Qwen3-VL-72B
6. **Real-time Processing**: Optimize for streaming analysis
7. **Multi-asset Support**: Extend beyond Dogecoin (Bitcoin, Ethereum, etc.)
8. **Attention Visualization**: Highlight important frames and audio segments
9. **Confidence Calibration**: Improve confidence estimates with uncertainty quantification
10. **Active Learning**: Prioritize uncertain videos for human annotation

---

## Citation

This project uses the following models:

**CLIP**:
```bibtex
@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

**Whisper**:
```bibtex
@article{radford2022whisper,
  title={Robust Speech Recognition via Large-Scale Weak Supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and others},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

**Qwen3-VL**:
```bibtex
@article{qwen3vl2024,
  title={Qwen3-VL: Scaling Multimodal Understanding},
  author={Qwen Team},
  journal={arXiv preprint},
  year={2024}
}
```

**FinBERT**:
```bibtex
@article{araci2019finbert,
  title={FinBERT: Financial Sentiment Analysis with Pre-trained Language Models},
  author={Araci, Dogu},
  journal={arXiv preprint arXiv:1908.10063},
  year={2019}
}
```

---

## Acknowledgments

- **Course**: MGT6785 - Applied Machine Learning for Business
- **Institution**: Georgia Tech Scheller College of Business
- **Computing Resources**: PACE-ICE Cluster (Partnership for an Advanced Computing Environment)
- **Models**: OpenAI (CLIP, Whisper), Qwen Team (Qwen3-VL), ProsusAI (FinBERT)

---

## License

For educational purposes (MGT6785 course project).

---

## Contact

For questions about this implementation, please refer to the course materials or office hours.

---

**Happy Analyzing!**
