# Cryptocurrency Video Sentiment Analysis with CLIP

This project analyzes cryptocurrency TikTok videos (specifically DogeCoin) to predict market sentiment using OpenAI's CLIP model.

## 📚 Table of Contents

- [Overview](#overview)
- [How CLIP Works for Video Sentiment Analysis](#how-clip-works-for-video-sentiment-analysis)
- [Technical Details](#technical-details)
- [Installation](#installation)
- [Usage](#usage)
- [Understanding the Results](#understanding-the-results)
- [Project Structure](#project-structure)

---

## Overview

This project implements a **CLIP-based video sentiment analyzer** that:
1. Extracts frames from cryptocurrency videos (TikTok, YouTube, etc.)
2. Uses OpenAI's CLIP model to understand visual content
3. Classifies sentiment as POSITIVE, NEGATIVE, or NEUTRAL
4. Generates sentiment scores for cryptocurrency market analysis

**No training required!** CLIP uses zero-shot learning to understand crypto concepts.

---

## How CLIP Works for Video Sentiment Analysis

### What is CLIP?

**CLIP (Contrastive Language-Image Pre-training)** is a neural network trained by OpenAI on 400 million image-text pairs from the internet. It understands the relationship between images and text descriptions.

### Why CLIP for TikTok Videos?

CLIP can "see" and understand visual cues in cryptocurrency videos:

- 📈 **Charts**: Price going up or down
- 😃 **Facial Expressions**: Happy, concerned, excited
- 📝 **Text Overlays**: "TO THE MOON!", "CRASH INCOMING"
- 🎨 **Colors**: Green (bullish), Red (bearish)
- 🚀 **Memes**: Rockets, moon references, Doge memes

### Zero-Shot Classification

CLIP doesn't need training on crypto videos specifically. It already understands:
- "bullish market" = positive sentiment
- "bearish market" = negative sentiment
- "growth", "moon", "pump" = positive
- "crash", "dump", "decline" = negative

### Our Approach Pipeline

```
📹 TikTok Video → Extract 5 Frames → For each frame:
                                     ↓
                              Feed to CLIP with labels:
                              - "positive crypto news, bullish, growth"
                              - "negative crypto news, bearish, crash"
                              - "neutral crypto discussion"
                                     ↓
                              CLIP outputs probabilities
                                     ↓
                              Sentiment = P(positive) - P(negative)
                                     ↓
                              Average across all 5 frames
```

### Real Example

If a TikTok shows:
- **Frame 1**: Person excited, pointing at rising chart → CLIP sees "bullish" 
- **Frame 2**: Green candles, text "TO THE MOON" → CLIP sees "positive growth"
- **Frame 3**: Elon Musk meme with rocket 🚀 → CLIP associates with "pump"
- **Frame 4**: Excited reactions → CLIP sees "enthusiasm"
- **Frame 5**: "BUY NOW!" overlay → CLIP sees "bullish signal"

**Result**: High positive sentiment score (e.g., 0.65)!

---

## Technical Details

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
- ✅ Green colors, rising charts
- ✅ Excited facial expressions
- ✅ Text overlays with positive words
- ✅ Rocket emojis, moon references 🚀🌙
- ✅ Bullish market imagery
- ✅ Memes and visual metaphors

### Complete Pipeline Visualization

```
🎬 TikTok Video: "DOGECOIN TO THE MOON! 🚀" (10 seconds)
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

- ✅ **Zero-shot learning**: No labeled training data needed
- ✅ **Visual understanding**: Analyzes actual video content, not just metadata
- ✅ **Meme comprehension**: Understands crypto memes and visual metaphors
- ✅ **Multi-frame analysis**: Captures video dynamics over time
- ✅ **Robust fallback**: Uses filename heuristics if frame extraction fails
- ✅ **Format agnostic**: Works with MP4, AVI, MOV, MKV, WebM

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch transformers opencv-python pandas numpy

# Optional: For GPU acceleration
# Install CUDA-enabled PyTorch from pytorch.org
```

### Verify Installation

Run the first few cells of `crypto.ipynb` to verify:
- ✓ PyTorch installed
- ✓ CUDA available (for GPU acceleration)
- ✓ OpenCV available
- ✓ Transformers available

---

## Usage

### 1. Prepare Your Videos

Place cryptocurrency videos in the `./videos/` directory with date-based names:

```
videos/
├── 2025-01-15.mp4
├── 2025-01-16.mp4
├── 20250117.mp4
└── 2025_01_18.mp4
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

### 2. Run the Analysis

Open `crypto.ipynb` and run all cells, or specifically run:

```python
# Initialize analyzer
analyzer = CLIPVideoSentimentAnalyzer()

# Analyze all videos
results_df = analyzer.batch_analyze_videos(
    video_dir="./videos",
    output_csv="./results/sentiment_analysis.csv"
)
```

### 3. View Results

Results are saved to `./results/sentiment_analysis.csv`:

```csv
date,video_path,sentiment_score,sentiment_class,confidence,method,num_frames_analyzed,timestamp
2025-01-15,videos/2025-01-15.mp4,0.65,POSITIVE,HIGH,CLIP visual analysis from 5 frames,5,2025-01-15T10:30:45
2025-01-16,videos/2025-01-16.mp4,-0.42,NEGATIVE,HIGH,CLIP visual analysis from 5 frames,5,2025-01-16T11:20:30
```

---

## Understanding the Results

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

### Output Columns

| Column | Description |
|--------|-------------|
| `date` | Extracted from filename (YYYY-MM-DD) |
| `video_path` | Path to the video file |
| `sentiment_score` | Numeric score from -1 to +1 |
| `sentiment_class` | POSITIVE, NEGATIVE, or NEUTRAL |
| `confidence` | HIGH, MEDIUM, or LOW |
| `method` | Analysis method used (CLIP visual analysis) |
| `num_frames_analyzed` | Number of frames successfully processed |
| `timestamp` | When the analysis was performed |

---

## Project Structure

```
MGT6785/
├── crypto.ipynb                  # Main analysis notebook
├── README.md                     # This file
├── videos/                       # Place your videos here
│   ├── 2025-01-15.mp4
│   └── 2025-01-16.mp4
├── results/                      # Output directory
│   └── sentiment_analysis.csv   # Analysis results
└── temp/                         # Temporary files (auto-created)
```

---

## Advanced Usage

### Adjust Frame Sampling

Extract more frames for longer videos:

```python
# Extract 10 frames instead of 5
frames = analyzer.extract_frames(video_path, num_frames=10)
```

### Custom Sentiment Labels

Modify labels in the `analyze_visual_sentiment_with_clip()` method:

```python
labels = [
    "extremely bullish cryptocurrency, to the moon, major pump",
    "extremely bearish cryptocurrency, major crash, dump",
    "neutral cryptocurrency discussion"
]
```

### Batch Processing with Progress

```python
for i, video_path in enumerate(video_files, 1):
    print(f"Processing {i}/{len(video_files)}: {video_path.name}")
    result = analyzer.analyze_video(str(video_path))
```

---

## Future Enhancements

To improve the analysis, consider adding:

1. **Audio Transcription** (Whisper): Analyze spoken content
2. **OCR Text Extraction** (EasyOCR): Extract text overlays
3. **Emotion Recognition** (FER): Detect facial expressions
4. **Multi-modal Fusion**: Combine visual + audio + text
5. **Larger CLIP Models**: Use `clip-vit-large-patch14` for better accuracy
6. **Fine-tuning**: Train on cryptocurrency-specific content
7. **Temporal Modeling**: Add LSTM/Transformer for frame sequences

---

## Troubleshooting

### No frames extracted

- Check video file integrity
- Verify OpenCV installation: `pip install opencv-python`
- Try different video format

### CLIP model not loading

- Install transformers: `pip install transformers`
- Check internet connection (for model download)
- Verify disk space (~1GB for CLIP model)

### Low GPU memory

- Reduce batch size or frame count
- Use CPU instead: `device='cpu'`
- Close other GPU-intensive applications

### Inconsistent results

- Videos may have mixed sentiment
- Try extracting more frames
- Check if videos contain actual crypto content

---

## Citation

This project uses:
- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **OpenAI CLIP**: `openai/clip-vit-base-patch32` from Hugging Face

---

## License

For educational purposes (MGT6785 course project).

---

## Contact

For questions about this implementation, please refer to the course materials or office hours.

---

**Happy Analyzing! 🚀📈**
