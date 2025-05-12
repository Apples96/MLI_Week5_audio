import os
import torch
import numpy as np
import librosa
import librosa.display
import torchaudio
from datasets import load_dataset
import pickle
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from collections import Counter


class UrbanSoundDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, max_length=1501):
            self.dataset = dataset
            self.max_length = max_length
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Get the spectrogram and convert to float32
            spectrogram = item['log_mel_spectrogram'].float()
            
            # Get original length (along time dimension)
            original_length = spectrogram.shape[1]
            
            # Create a tensor to hold the padded/clipped spectrogram
            padded_spectrogram = torch.zeros(spectrogram.shape[0], self.max_length, dtype=torch.float32)
            
            if original_length >= self.max_length:
                # If the original is longer than max_length, clip it
                padded_spectrogram = spectrogram[:, :self.max_length]
            else:
                # If the original is shorter, copy it and leave the rest as zeros (padding)
                padded_spectrogram[:, :original_length] = spectrogram
            
            return {
                'log_mel_spectrogram': padded_spectrogram,
                'classID': torch.tensor(item['classID'], dtype=torch.long),
                'fold': item['fold'],
                'original_length': torch.tensor(original_length, dtype=torch.long)  # Store original length for reference
            }


def download_process_save_urbansound8K_dataset(root_dir = "./data"):
    # Create data directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)
    
    # Load dataset from Hugging Face and save to disk
    # This is more efficient than downloading every time
    dataset = load_dataset("danavery/urbansound8K")
    processed_dataset = []
    
    # Use tqdm for progress visualization
    for idx, item in enumerate(tqdm(dataset['train'], desc="Processing audio files")):
        # Convert audio to mel spectrogram
        audio_array = np.array(item["audio"]["array"])
        
        # Generate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_array, 
            sr=22050,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # Convert to log mel spectrogram
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        
        # Convert to tensor
        log_mel_spectrogram = torch.tensor(log_mel_spectrogram)
        
        # Create dictionary with processed data and metadata
        processed_item = {
            "log_mel_spectrogram": log_mel_spectrogram,
            "slice_file_name": item["slice_file_name"],
            "fsID": item["fsID"],
            "start": item["start"],
            "end": item["end"],
            "salience": item["salience"],
            "fold": item["fold"],
            "classID": item["classID"],
            "class": item["class"]
        }
        
        processed_dataset.append(processed_item)
    
    # Save processed dataset as pickle file
    output_path = os.path.join(root_dir, "processed_urbansound8k.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(processed_dataset, f)
    
    # Print length of processed dataset
    print(f"Length of processed dataset: {len(processed_dataset)} samples")
    
    # Print storage size of processed dataset
    storage_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    print(f"Storage size of processed dataset: {storage_size:.2f} MB")
    
    return processed_dataset


def load_and_visualize_dataset(file_path="./data/processed_urbansound8k.pkl"):
    """
    Load the processed UrbanSound8K dataset and visualize examples and statistics.
    
    Args:
        file_path: Path to the pickle file containing the processed dataset
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    # Load the dataset
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Print basic statistics
    print(f"\nDataset Statistics:")
    print(f"Number of samples: {len(dataset)}")
    
    # Get class distribution
    class_counts = Counter([item['class'] for item in dataset])
    print("\nClass Distribution:")
    for cls, count in class_counts.most_common():
        print(f"  {cls}: {count} samples")
    
    # Print spectrogram shapes
    spectrogram_shapes = [item['log_mel_spectrogram'].shape for item in dataset]
    unique_shapes = set(spectrogram_shapes)
    max_length = max(shape[1] for shape in spectrogram_shapes)
    min_length = min(shape[1] for shape in spectrogram_shapes)
    print(f"\nUnique spectrogram shapes: {unique_shapes}")
    print(f"Max spectrogram length (time dimension): {max_length} frames")
    print(f"Min spectrogram length (time dimension): {min_length} frames")
    
    # Calculate mean duration
    durations = [(item['end'] - item['start']) for item in dataset]
    mean_duration = np.mean(durations)
    min_duration = np.min(durations)
    max_duration = np.max(durations)
    print(f"\nAudio Duration Statistics:")
    print(f"  Mean: {mean_duration:.2f} seconds")
    print(f"  Min: {min_duration:.2f} seconds")
    print(f"  Max: {max_duration:.2f} seconds")
    
    # Visualize some examples
    print("\nVisualizing random examples...")
    
    # Randomly select 3 examples
    random_indices = np.random.choice(len(dataset), size=3, replace=False)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(random_indices):
        item = dataset[idx]
        
        # Convert tensor to numpy if needed
        if isinstance(item['log_mel_spectrogram'], torch.Tensor):
            spectrogram = item['log_mel_spectrogram'].numpy()
        else:
            spectrogram = item['log_mel_spectrogram']
        
        plt.subplot(3, 1, i+1)
        librosa.display.specshow(
            spectrogram,
            x_axis='time',
            y_axis='mel',
            sr=22050,  # Default sampling rate for UrbanSound8K
            hop_length=512
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Class: {item['class']} (ID: {item['classID']})")
    
    plt.tight_layout()
    plt.savefig("./data/example_spectrograms.png")
    print(f"Example spectrograms saved to ./data/example_spectrograms.png")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    classes = [cls for cls, _ in class_counts.most_common()]
    counts = [count for _, count in class_counts.most_common()]
    
    plt.bar(classes, counts)
    plt.title('Class Distribution in UrbanSound8K Dataset')
    plt.xlabel('Sound Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./data/class_distribution.png")
    print(f"Class distribution plot saved to ./data/class_distribution.png")




if __name__ == "__main__":
    download_process_save_urbansound8K_dataset()
    dataset = load_and_visualize_dataset()
    print(f"\nExample of a single item's metadata (excluding spectrogram):")
    if dataset:
        # Show example of metadata (excluding the large spectrogram)
        example = dataset[0].copy()
        example.pop('log_mel_spectrogram')
        for key, value in example.items():
            print(f"  {key}: {value}")