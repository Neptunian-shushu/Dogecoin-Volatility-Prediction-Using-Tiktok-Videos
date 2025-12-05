#!/usr/bin/env python3
"""
Qwen2.5-Omni LoRA Fine-tuning on MEAD Emotion Dataset
Fine-tunes multimodal model for crypto sentiment analysis
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ============================================================================
# Configuration
# ============================================================================
MEAD_ROOT = "./data/mead"
OUTPUT_DIR = "./checkpoints/mead_finetuned"
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"

# Training hyperparameters
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUMULATION = 16
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32

# ============================================================================
# Dataset
# ============================================================================
class MEADEmotionDataset(Dataset):
    """MEAD Dataset for emotion fine-tuning with video+audio"""
    
    EMOTION_TO_SENTIMENT = {
        'angry': 'negative', 'contempt': 'negative', 'disgusted': 'negative',
        'fear': 'negative', 'happy': 'positive', 'neutral': 'neutral',
        'sad': 'negative', 'surprised': 'neutral'
    }
    SENTIMENT_TO_ID = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def __init__(self, mead_root: str, processor, split: str = 'train'):
        self.mead_root = Path(mead_root)
        self.processor = processor
        self.split = split
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self):
        samples = []
        actor_dirs = sorted([d for d in self.mead_root.iterdir() 
                           if d.is_dir() and d.name.startswith('M')])
        
        for actor_dir in actor_dirs:
            video_root = actor_dir / "video" / "down"
            audio_root = actor_dir / "audio"
            
            if not (video_root.exists() and audio_root.exists()):
                continue
            
            for emotion in self.EMOTION_TO_SENTIMENT.keys():
                video_path = video_root / emotion
                audio_path = audio_root / emotion
                
                if not (video_path.exists() and audio_path.exists()):
                    continue
                
                for level_dir in video_path.glob("level_*"):
                    audio_level = audio_path / level_dir.name
                    if not audio_level.exists():
                        continue
                    
                    for video_file in level_dir.glob("*.mp4"):
                        audio_file = audio_level / f"{video_file.stem}.m4a"
                        if audio_file.exists():
                            samples.append({
                                'video_path': video_file,
                                'audio_path': audio_file,
                                'emotion': emotion,
                                'sentiment_id': self.SENTIMENT_TO_ID[self.EMOTION_TO_SENTIMENT[emotion]]
                            })
        
        # Split train/val
        split_idx = int(len(samples) * 0.8)
        return samples[:split_idx] if self.split == 'train' else samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = f"Analyze the emotion in this video. The person shows {sample['emotion']} emotion."
        
        # Process video and audio separately (MEAD has separate .mp4 and .m4a files)
        inputs = self.processor(
            text=prompt,
            videos=str(sample['video_path']),
            audios=str(sample['audio_path']),
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in inputs.items()}
        inputs['label'] = sample['sentiment_id']
        
        return inputs

# ============================================================================
# Data Collator
# ============================================================================
class MultimodalDataCollator:
    """Custom collator for variable-length video+audio"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        batch = {}
        
        for key in features[0].keys():
            if key == 'label':
                continue
            
            if isinstance(features[0][key], torch.Tensor):
                tensors = [f[key] for f in features]
                shapes = [t.shape for t in tensors]
                
                if len(set(shapes)) == 1:
                    batch[key] = torch.stack(tensors)
                else:
                    # Pad to max shape
                    max_shape = [max(t.shape[i] for t in tensors) for i in range(tensors[0].ndim)]
                    padded = []
                    for t in tensors:
                        padding = []
                        for i in range(t.ndim - 1, -1, -1):
                            padding.extend([0, max_shape[i] - t.shape[i]])
                        padded.append(torch.nn.functional.pad(t, padding, value=0))
                    batch[key] = torch.stack(padded)
        
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=torch.long)
        return batch

# ============================================================================
# Custom Trainer
# ============================================================================
class LossPlottingCallback(TrainerCallback):
    """Callback to plot and save loss curves during training"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
        os.makedirs(output_dir, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
            
            # Plot every 20 steps
            if len(self.train_losses) > 0 and state.global_step % 20 == 0:
                self._plot_losses()
    
    def _plot_losses(self):
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        if len(self.train_losses) > 0:
            plt.plot(self.steps, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss
        if len(self.eval_losses) > 0:
            plt.plot(self.eval_steps, self.eval_losses, 'r-', label='Validation Loss', linewidth=2, marker='o')
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Qwen2.5-Omni LoRA Fine-tuning Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{self.output_dir}/loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save loss data to CSV
        if len(self.train_losses) > 0:
            pd.DataFrame({
                'step': self.steps,
                'train_loss': self.train_losses
            }).to_csv(f'{self.output_dir}/train_losses.csv', index=False)
        
        if len(self.eval_losses) > 0:
            pd.DataFrame({
                'step': self.eval_steps,
                'eval_loss': self.eval_losses
            }).to_csv(f'{self.output_dir}/eval_losses.csv', index=False)

class MultimodalTrainer(Trainer):
    """Custom trainer for Qwen2.5-Omni multimodal inputs"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        # Remove input_ids if present (not used by Qwen2.5-Omni)
        inputs.pop('input_ids', None)
        
        # Unwrap all model layers to get base model
        base_model = model
        while hasattr(base_model, 'base_model') or hasattr(base_model, 'model'):
            base_model = base_model.base_model if hasattr(base_model, 'base_model') else base_model.model
        
        outputs = base_model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# Main Training
# ============================================================================
def main():
    print("="*80)
    print("QWEN2.5-OMNI LORA FINE-TUNING ON MEAD")
    print("="*80)
    
    # Load model and processor
    print("\n1. Loading model...")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Apply LoRA (without prepare_model_for_kbit_training since we're not using quantization)
    print("\n2. Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
    
    # Load datasets
    print("\n3. Loading MEAD dataset...")
    train_dataset = MEADEmotionDataset(MEAD_ROOT, processor, split='train')
    val_dataset = MEADEmotionDataset(MEAD_ROOT, processor, split='val')
    
    # Training arguments
    print("\n4. Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="no",
        fp16=False,
        bf16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    
    # Create trainer
    data_collator = MultimodalDataCollator(processor)
    loss_callback = LossPlottingCallback(OUTPUT_DIR)
    
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )
    
    # Train
    print("\n5. Starting training...")
    print(f"   Total steps: ~{len(train_dataset) // GRAD_ACCUMULATION * NUM_EPOCHS}")
    print(f"   Loss curves will be saved to: {OUTPUT_DIR}/loss_curve.png")
    
    train_result = trainer.train()
    
    # Save
    print("\n6. Saving model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    print(f"  Model saved: {OUTPUT_DIR}/final")
    print(f"  Loss curve: {OUTPUT_DIR}/loss_curve.png")
    print(f"  Loss data: {OUTPUT_DIR}/train_losses.csv, {OUTPUT_DIR}/eval_losses.csv")
    print("="*80)

if __name__ == "__main__":
    main()
