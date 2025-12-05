#!/usr/bin/env python3
"""
Preprocess MEAD videos for faster training
Compress videos: reduce resolution, frame rate, and duration
This is a ONE-TIME preprocessing step that will make training much faster
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
MEAD_ROOT = Path("./data/mead")
OUTPUT_ROOT = Path("./data/mead_compressed")
TARGET_FPS = 5  # Reduce from ~30 fps to 5 fps (6x reduction)
TARGET_RESOLUTION = (224, 224)  # CLIP/vision models use 224x224
MAX_DURATION_SECONDS = 3  # Limit to 3 seconds (emotions are short)
QUALITY = 85  # JPEG quality for frames (0-100)

def get_video_info(video_path):
    """Get video metadata"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, duration, width, height

def compress_video(input_path, output_path, target_fps=5, resolution=(224, 224), max_duration=3):
    """
    Compress video by:
    1. Reducing resolution to 224x224
    2. Reducing frame rate to 5 fps
    3. Limiting duration to 3 seconds
    """
    cap = cv2.VideoCapture(str(input_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if original_fps == 0:
        print(f"   ⚠️ Warning: Could not read FPS from {input_path.name}")
        cap.release()
        return False
    
    # Calculate frame skip
    frame_skip = max(1, int(original_fps / target_fps))
    max_frames = int(target_fps * max_duration)
    
    # Prepare output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, resolution)
    
    frame_count = 0
    written_frames = 0
    
    while written_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames at target fps
        if frame_count % frame_skip == 0:
            # Resize
            frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            out.write(frame)
            written_frames += 1
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    return written_frames > 0

def compress_audio(input_path, output_path):
    """
    Copy audio (or could downsample if needed)
    For now, just copy the audio file
    """
    shutil.copy2(input_path, output_path)
    return True

def preprocess_mead_dataset():
    """Preprocess entire MEAD dataset"""
    print("="*80)
    print("MEAD DATASET PREPROCESSING")
    print("="*80)
    print(f"Input:  {MEAD_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Settings:")
    print(f"  - Target FPS: {TARGET_FPS}")
    print(f"  - Resolution: {TARGET_RESOLUTION}")
    print(f"  - Max duration: {MAX_DURATION_SECONDS}s")
    print("="*80)
    
    # Find all videos
    video_files = list(MEAD_ROOT.rglob("*.mp4"))
    audio_files = list(MEAD_ROOT.rglob("*.m4a"))
    
    print(f"\nFound {len(video_files)} videos and {len(audio_files)} audio files")
    
    # Process videos
    print("\n📹 Processing videos...")
    video_stats = {'original_size': 0, 'compressed_size': 0, 'success': 0, 'failed': 0}
    
    for video_file in tqdm(video_files, desc="Compressing videos"):
        # Create output path
        relative_path = video_file.relative_to(MEAD_ROOT)
        output_file = OUTPUT_ROOT / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get original size
        original_size = video_file.stat().st_size
        video_stats['original_size'] += original_size
        
        # Compress
        success = compress_video(
            video_file, 
            output_file,
            target_fps=TARGET_FPS,
            resolution=TARGET_RESOLUTION,
            max_duration=MAX_DURATION_SECONDS
        )
        
        if success:
            compressed_size = output_file.stat().st_size
            video_stats['compressed_size'] += compressed_size
            video_stats['success'] += 1
        else:
            video_stats['failed'] += 1
    
    # Process audio files
    print("\n🔊 Processing audio files...")
    audio_stats = {'success': 0, 'failed': 0}
    
    for audio_file in tqdm(audio_files, desc="Copying audio"):
        # Create output path
        relative_path = audio_file.relative_to(MEAD_ROOT)
        output_file = OUTPUT_ROOT / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy (or downsample if needed)
        try:
            compress_audio(audio_file, output_file)
            audio_stats['success'] += 1
        except Exception as e:
            print(f"   ⚠️ Failed: {audio_file.name}: {e}")
            audio_stats['failed'] += 1
    
    # Print statistics
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    
    print("\n📊 Video Statistics:")
    print(f"   Total videos: {len(video_files)}")
    print(f"   Successfully compressed: {video_stats['success']}")
    print(f"   Failed: {video_stats['failed']}")
    print(f"   Original size: {video_stats['original_size'] / 1e9:.2f} GB")
    print(f"   Compressed size: {video_stats['compressed_size'] / 1e9:.2f} GB")
    compression_ratio = (1 - video_stats['compressed_size'] / video_stats['original_size']) * 100
    print(f"   Compression ratio: {compression_ratio:.1f}% reduction")
    
    print("\n🔊 Audio Statistics:")
    print(f"   Total audio files: {len(audio_files)}")
    print(f"   Successfully copied: {audio_stats['success']}")
    print(f"   Failed: {audio_stats['failed']}")
    
    print("\n✅ Next steps:")
    print(f"   1. Update train_qwen_omni_fast.py:")
    print(f"      MEAD_ROOT = \"{OUTPUT_ROOT}\"")
    print(f"   2. Run training with compressed videos")
    print(f"   3. Expected speedup: {compression_ratio / 20:.1f}x faster video loading")
    print("="*80)

def test_single_video():
    """Test compression on a single video"""
    print("Testing single video compression...")
    
    # Find first video
    video_files = list(MEAD_ROOT.rglob("*.mp4"))
    if not video_files:
        print("No videos found!")
        return
    
    test_video = video_files[0]
    print(f"\nTest video: {test_video.name}")
    
    # Get info
    fps, frames, duration, width, height = get_video_info(test_video)
    print(f"Original: {width}x{height}, {fps:.1f} fps, {duration:.2f}s, {frames} frames")
    print(f"Size: {test_video.stat().st_size / 1e6:.2f} MB")
    
    # Compress
    output_test = OUTPUT_ROOT / "test_compressed.mp4"
    output_test.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nCompressing...")
    compress_video(test_video, output_test, TARGET_FPS, TARGET_RESOLUTION, MAX_DURATION_SECONDS)
    
    # Get compressed info
    fps_c, frames_c, duration_c, width_c, height_c = get_video_info(output_test)
    print(f"Compressed: {width_c}x{height_c}, {fps_c:.1f} fps, {duration_c:.2f}s, {frames_c} frames")
    print(f"Size: {output_test.stat().st_size / 1e6:.2f} MB")
    
    compression = (1 - output_test.stat().st_size / test_video.stat().st_size) * 100
    print(f"\nCompression: {compression:.1f}% reduction")
    print(f"Speedup estimate: {frames / frames_c:.1f}x faster to process")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_single_video()
    else:
        preprocess_mead_dataset()
