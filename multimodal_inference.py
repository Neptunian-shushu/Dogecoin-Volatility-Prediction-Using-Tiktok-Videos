#!/usr/bin/env python3
"""
Multimodal Fusion Sentiment Analysis for Cryptocurrency Videos
Optimized for PACE-ICE Cluster Execution

This script performs sentiment analysis on TikTok videos using:
- Audio Branch: Whisper + FinBERT
- Visual Branch: CLIP
- Reasoning Branch: Qwen3-VL
- Fusion: Weighted combination
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import cv2
from typing import Dict, List, Tuple
import re
import subprocess
from PIL import Image
import argparse

warnings.filterwarnings('ignore')

# Import model-specific libraries
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    CLIPProcessor,
    CLIPModel,
    pipeline
)
from qwen_vl_utils import process_vision_info
import whisper


class MultimodalSentimentAnalyzer:
    """Multimodal sentiment analyzer for cryptocurrency videos"""
    
    def __init__(self, device=None):
        """Initialize all models"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self._load_models()
        
    def _load_models(self):
        """Load all required models"""
        print("="*80)
        print("LOADING ALL MODELS")
        print("="*80)
        
        # 1. Whisper
        print("\n1. Loading Whisper (base)...")
        self.whisper_model = whisper.load_model("base")
        print("   ✓ Whisper loaded")
        
        # 2. CLIP
        print("\n2. Loading CLIP (ViT-B/32)...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = self.clip_model.to(self.device)
        print(f"   ✓ CLIP loaded on {self.device}")
        
        # 3. Qwen3-VL
        print("\n3. Loading Qwen3-VL-8B-Instruct...")
        self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.qwen_model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"   ✓ Qwen3-VL loaded")
        
        # 4. FinBERT
        print("\n4. Loading FinBERT...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )
        print("   ✓ FinBERT loaded")
        
        print("\n" + "="*80)
        print("ALL MODELS LOADED SUCCESSFULLY")
        print("="*80 + "\n")
    
    def extract_audio_and_frames(self, video_path: str, fps: float = 1.0) -> Tuple[str, List[np.ndarray]]:
        """Extract audio and frames from video"""
        video_path = Path(video_path)
        audio_path = f"./temp/{video_path.stem}_audio.wav"
        
        # Extract audio
        try:
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", "-y",
                audio_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except:
            audio_path = None
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / fps))
        
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        
        cap.release()
        return audio_path, frames
    
    def audio_branch(self, audio_path: str) -> Dict:
        """Process audio: Whisper + FinBERT"""
        if not audio_path or not os.path.exists(audio_path):
            return {'transcript': "", 'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
        
        # Transcribe
        try:
            result = self.whisper_model.transcribe(audio_path)
            transcript = result['text'].strip()
        except:
            return {'transcript': "", 'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
        
        if not transcript or len(transcript) < 10:
            return {'transcript': transcript, 'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'NEUTRAL'}
        
        # Sentiment analysis
        transcript_truncated = transcript[:512]
        sentiment_results = self.sentiment_pipeline(transcript_truncated, top_k=None)
        
        if isinstance(sentiment_results[0], list):
            sentiment_results = sentiment_results[0]
        
        probs = {item['label'].lower(): item['score'] for item in sentiment_results}
        
        # Adaptive scaling based on neutral probability
        pos_prob = probs.get('positive', 0)
        neg_prob = probs.get('negative', 0)
        neutral_prob = probs.get('neutral', 0)
        
        raw_score = pos_prob - neg_prob
        amplification_strength = 10
        scale_factor = 1.0 + (neutral_prob * amplification_strength)
        sentiment_score = np.clip(raw_score * scale_factor, -1.0, 1.0)
        
        label = max(probs.items(), key=lambda x: x[1])[0].upper()
        confidence = max(probs.values())
        
        return {
            'transcript': transcript,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'label': label
        }
    
    def visual_branch(self, frames: List[np.ndarray]) -> Dict:
        """Process visual: CLIP embeddings"""
        if not frames:
            return {'sentiment_score': 0.0, 'confidence': 0.0, 'embeddings': None}
        
        # Get embeddings
        pil_frames = [Image.fromarray(frame) for frame in frames]
        embeddings = []
        
        for frame in pil_frames:
            inputs = self.clip_processor(images=frame, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                embeddings.append(image_features.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        v_video = np.mean(embeddings, axis=0)
        
        # Sentiment classification
        sentiment_texts = [
            "positive cryptocurrency news, bullish market, growth, moon, pump",
            "negative cryptocurrency news, bearish market, decline, crash, dump",
            "neutral cryptocurrency discussion, stable market, sideways"
        ]
        
        text_inputs = self.clip_processor(text=sentiment_texts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features.cpu().numpy()
        
        similarities = np.dot(text_features, v_video.T).flatten()
        similarities = np.exp(similarities) / np.sum(np.exp(similarities))
        
        sentiment_score = similarities[0] - similarities[1]
        confidence = max(similarities)
        
        return {
            'sentiment_score': float(sentiment_score),
            'confidence': float(confidence),
            'embeddings': embeddings,
            'similarities': similarities.tolist()
        }
    
    def select_key_frames(self, frames: List[np.ndarray], embeddings: np.ndarray, 
                         num_key_frames: int = 3) -> List[np.ndarray]:
        """Select diverse key frames"""
        if len(frames) <= num_key_frames:
            return frames
        
        selected_indices = [0]
        remaining_indices = list(range(1, len(frames)))
        
        for _ in range(num_key_frames - 1):
            max_min_distance = -1
            best_idx = None
            
            for idx in remaining_indices:
                distances = [np.linalg.norm(embeddings[idx] - embeddings[sel_idx])
                           for sel_idx in selected_indices]
                min_distance = min(distances)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        selected_indices.sort()
        return [frames[i] for i in selected_indices]
    
    def reasoning_branch(self, transcript: str, key_frames: List[np.ndarray]) -> Dict:
        """Process reasoning: Qwen3-VL multimodal analysis"""
        if not transcript or len(transcript) < 10:
            transcript = "[No clear audio content detected]"
        
        # Save key frames
        temp_frame_paths = []
        for i, frame in enumerate(key_frames):
            temp_path = f"./temp/key_frame_{i}.jpg"
            Image.fromarray(frame).save(temp_path)
            temp_frame_paths.append(temp_path)
        
        # Create prompt
        prompt = f"""You are an expert financial analyst specializing in cryptocurrency sentiment analysis. Analyze this TikTok video about cryptocurrency/Dogecoin using both the transcript and visual content.

TRANSCRIPT:
{transcript}

VISUAL CONTENT:
You are provided with {len(key_frames)} key frames from the video.

INSTRUCTIONS:
Analyze the overall sentiment of this video considering:
- Bullish signals: "pump", "moon", "buy", "rocket", "gains", "profit", "bullish", "surge", excitement, positive visuals
- Bearish signals: "dump", "crash", "sell", "bearish", "loss", "drop", "falling", worry, negative visuals
- Neutral signals: "hold", "wait", "uncertain", "sideways", "stable", neutral tone

Provide your analysis in this EXACT format:

SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
SCORE: [number from -1.0 to +1.0, where -1.0 is extremely bearish, 0 is neutral, +1.0 is extremely bullish]
REASONING: [Brief explanation of your assessment combining audio and visual insights]
"""
        
        # Prepare messages
        content = [{"type": "text", "text": prompt}]
        for frame_path in temp_frame_paths:
            content.append({"type": "image", "image": frame_path})
        
        messages = [{"role": "user", "content": content}]
        
        # Generate
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.qwen_model.device)
        
        with torch.no_grad():
            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] 
                                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse response
        result = self._parse_qwen_response(response)
        
        # Cleanup
        for path in temp_frame_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return result
    
    def _parse_qwen_response(self, response: str) -> Dict:
        """Parse Qwen3-VL response"""
        sentiment_class = 'NEUTRAL'
        confidence = 'MEDIUM'
        score = 0.0
        reasoning = response
        
        # Extract fields
        if 'SENTIMENT:' in response:
            sentiment_match = re.search(r'SENTIMENT:\s*(\w+)', response, re.IGNORECASE)
            if sentiment_match:
                sentiment_class = sentiment_match.group(1).upper()
        
        if 'CONFIDENCE:' in response:
            confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', response, re.IGNORECASE)
            if confidence_match:
                confidence = confidence_match.group(1).upper()
        
        if 'SCORE:' in response:
            score_match = re.search(r'SCORE:\s*([+-]?\d+\.?\d*)', response, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(-1.0, min(1.0, score))
                except:
                    pass
        
        # Infer score if needed
        if score == 0.0 and sentiment_class != 'NEUTRAL':
            confidence_map = {'HIGH': 0.8, 'MEDIUM': 0.5, 'LOW': 0.3}
            magnitude = confidence_map.get(confidence, 0.5)
            
            if 'POSITIVE' in sentiment_class:
                score = magnitude
            elif 'NEGATIVE' in sentiment_class:
                score = -magnitude
        
        if 'REASONING:' in response:
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
        
        return {
            'sentiment_score': score,
            'confidence': confidence,
            'reasoning': reasoning,
            'sentiment_class': sentiment_class
        }
    
    def fusion(self, audio_result: Dict, visual_result: Dict, reasoning_result: Dict,
               weights: Dict = {'audio': 0.1, 'visual': 0.2, 'reasoning': 0.7}) -> Dict:
        """Fuse results from all branches"""
        audio_score = audio_result['sentiment_score']
        visual_score = visual_result['sentiment_score']
        reasoning_score = reasoning_result['sentiment_score']
        
        final_score = (
            weights['audio'] * audio_score +
            weights['visual'] * visual_score +
            weights['reasoning'] * reasoning_score
        )
        
        if final_score > 0.2:
            final_class = 'POSITIVE'
        elif final_score < -0.2:
            final_class = 'NEGATIVE'
        else:
            final_class = 'NEUTRAL'
        
        return {
            'final_sentiment_score': float(final_score),
            'final_sentiment_class': final_class,
            'branch_scores': {
                'audio': float(audio_score),
                'visual': float(visual_score),
                'reasoning': float(reasoning_score)
            },
            'branch_weights': weights,
            'components': {
                'audio': audio_result,
                'visual': visual_result,
                'reasoning': reasoning_result
            }
        }
    
    def analyze_video(self, video_path: str, 
                     fusion_weights: Dict = {'audio': 0.1, 'visual': 0.2, 'reasoning': 0.7}) -> Dict:
        """Complete analysis pipeline for a single video"""
        print(f"\nAnalyzing: {Path(video_path).name}")
        
        # Extract
        audio_path, frames = self.extract_audio_and_frames(video_path, fps=1.0)
        
        # Audio branch
        audio_result = self.audio_branch(audio_path)
        
        # Visual branch
        visual_result = self.visual_branch(frames)
        
        # Key frames
        key_frames = self.select_key_frames(frames, visual_result['embeddings'], num_key_frames=3)
        
        # Reasoning branch
        reasoning_result = self.reasoning_branch(audio_result['transcript'], key_frames)
        
        # Fusion
        final_result = self.fusion(audio_result, visual_result, reasoning_result, fusion_weights)
        
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        print(f"✓ Score: {final_result['final_sentiment_score']:.3f} ({final_result['final_sentiment_class']})")
        
        return final_result
    
    def process_all_videos(self, video_dir: str, output_file: str, details_file: str,
                          fusion_weights: Dict = {'audio': 0.1, 'visual': 0.2, 'reasoning': 0.7}):
        """Process all videos in directory"""
        video_files = list(Path(video_dir).glob("*.mp4"))
        print(f"\nFound {len(video_files)} videos to process\n")
        
        results = []
        detailed_results = []
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
            
            try:
                result = self.analyze_video(str(video_path), fusion_weights)
                
                # Extract date
                date_match = re.search(r'(\d{4})[_-](\d{2})[_-](\d{2})', video_path.name)
                date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else video_path.stem
                
                # Store results
                row = {
                    'date': date,
                    'video_path': video_path.name,
                    'final_sentiment_score': result['final_sentiment_score'],
                    'final_sentiment_class': result['final_sentiment_class'],
                    'audio_score': result['branch_scores']['audio'],
                    'visual_score': result['branch_scores']['visual'],
                    'reasoning_score': result['branch_scores']['reasoning'],
                    'method': 'Multimodal Fusion (Whisper + CLIP + Qwen3-VL)',
                    'timestamp': datetime.now().isoformat()
                }
                results.append(row)
                
                detailed_row = {
                    'date': date,
                    'video_path': video_path.name,
                    'transcript': result['components']['audio']['transcript'],
                    'reasoning': result['components']['reasoning']['reasoning'],
                    'timestamp': datetime.now().isoformat()
                }
                detailed_results.append(detailed_row)
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                continue
        
        # Save results
        if results:
            # Detailed results
            detailed_df = pd.DataFrame(detailed_results)
            detailed_df = detailed_df.sort_values('date')
            detailed_df.to_csv(details_file, index=False)
            print(f"\n✓ Detailed results saved: {details_file}")
            
            # Consolidated by date
            df = pd.DataFrame(results)
            df = df.sort_values('date')
            
            aggregated = df.groupby('date').agg({
                'final_sentiment_score': 'mean',
                'audio_score': 'mean',
                'visual_score': 'mean',
                'reasoning_score': 'mean',
                'video_path': lambda x: '; '.join(x),
                'method': 'first',
                'timestamp': 'first'
            }).reset_index()
            
            def classify_sentiment(score):
                return 'POSITIVE' if score > 0.2 else ('NEGATIVE' if score < -0.2 else 'NEUTRAL')
            
            aggregated['final_sentiment_class'] = aggregated['final_sentiment_score'].apply(classify_sentiment)
            aggregated['num_videos'] = df.groupby('date').size().values
            
            aggregated = aggregated[['date', 'num_videos', 'video_path', 'final_sentiment_score', 
                                    'final_sentiment_class', 'audio_score', 'visual_score', 
                                    'reasoning_score', 'method', 'timestamp']]
            
            aggregated.to_csv(output_file, index=False)
            
            print(f"✓ Sentiment scores saved: {output_file}")
            print(f"\n{'='*80}")
            print(f"COMPLETE: Processed {len(df)} videos across {len(aggregated)} unique dates")
            print(f"{'='*80}")
            
            return aggregated
        else:
            print("\n✗ No results to save")
            return pd.DataFrame()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis for Cryptocurrency Videos')
    parser.add_argument('--video_dir', type=str, default='./videos', help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--audio_weight', type=float, default=0.1, help='Weight for audio branch')
    parser.add_argument('--visual_weight', type=float, default=0.2, help='Weight for visual branch')
    parser.add_argument('--reasoning_weight', type=float, default=0.7, help='Weight for reasoning branch')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    
    # Initialize analyzer
    print("\nInitializing Multimodal Sentiment Analyzer...")
    analyzer = MultimodalSentimentAnalyzer()
    
    # Process videos
    fusion_weights = {
        'audio': args.audio_weight,
        'visual': args.visual_weight,
        'reasoning': args.reasoning_weight
    }
    
    output_file = os.path.join(args.output_dir, 'multimodal_fusion_sentiment.csv')
    details_file = os.path.join(args.output_dir, 'multimodal_fusion_details.csv')
    
    analyzer.process_all_videos(
        video_dir=args.video_dir,
        output_file=output_file,
        details_file=details_file,
        fusion_weights=fusion_weights
    )


if __name__ == "__main__":
    main()
