import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from model import UrbanSoundClassifier  # Import your model class
from data import UrbanSoundDataset



def get_data_loader(file_path="./data/processed_urbansound8k.pkl", batch_size=32, test_fold=10):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    # Load the dataset
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    full_dataset = UrbanSoundDataset(dataset)

    # Split into train and test by fold
    train_indices = [i for i in range(len(full_dataset)) if full_dataset[i]['fold'] != test_fold]
    test_indices = [i for i in range(len(full_dataset)) if full_dataset[i]['fold'] == test_fold]
   

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"Created train dataloader with {len(train_dataset)} samples from folds 1-{test_fold-1} and {test_fold+1}-10")
    print(f"Created test dataloader with {len(test_dataset)} samples from fold {test_fold}")
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader




def train_urban_sound_classifier():
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Perform 10-fold cross validation
    all_accuracies = []
    
    for test_fold in range(1, 11):  # 10-fold cross validation
        print(f"\n---------- FOLD {test_fold} ----------")
        
        # Get dataloaders for this fold
        train_dataset, test_dataset, train_dataloader, test_dataloader = get_data_loader(test_fold=test_fold)
        if train_dataset is None:
            continue
        
        # Initialize model, loss function, and optimizer
        # Determine max sequence length from dataset
        max_seq_length = 1501  # Using the known maximum from the dataset
        
        model = UrbanSoundClassifier(max_seq_length=max_seq_length)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)

        # Training
        epochs = 3
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            
            for idx, batch in enumerate(train_dataloader):
                # Move data to device
                log_mel_spectrogram = batch['log_mel_spectrogram'].to(device)
                classID = batch['classID'].to(device)
                
                optimizer.zero_grad()
                logits = model(log_mel_spectrogram)  
                loss = criterion(logits, classID)
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Print stats 
                if idx % 10 == 0:  # Show update every 10 batches
                    print(f'Fold {test_fold}, Epoch {epoch} [{idx * len(log_mel_spectrogram)}/{len(train_dataset)} '
                          f'({100. * idx * len(log_mel_spectrogram) / len(train_dataset):.0f}%)]\tLoss: {loss.item():.6f}')
            
            avg_train_loss = epoch_train_loss / len(train_dataloader)

            model_path = f'models/urban_sound_classifier_fold{test_fold}.pt'
            torch.save(model.state_dict(), model_path)

            # Test model after each epoch
            model.eval()
            epoch_test_loss = 0
            correct = 0
            with torch.no_grad():  # No need to track gradients during evaluation
                for batch in test_dataloader:
                    # Move data to device
                    log_mel_spectrogram = batch['log_mel_spectrogram'].to(device)
                    classID = batch['classID'].to(device)
                    
                    logits = model(log_mel_spectrogram)
                    loss = criterion(logits, classID)
                    epoch_test_loss += loss.item()
                    
                    # Get predictions
                    pred = logits.argmax(dim=1)
                    correct += (pred == classID).sum().item()
            
            avg_test_loss = epoch_test_loss / len(test_dataloader)
            accuracy = 100. * correct / len(test_dataset)
        
            print(f'Fold {test_fold}, Epoch {epoch}: '
                  f'Train loss: {avg_train_loss:.4f}, '
                  f'Test loss: {avg_test_loss:.4f}, '
                  f'Accuracy: {correct}/{len(test_dataset)} '
                  f'({accuracy:.2f}%)')
        
        # Save the final fold results
        all_accuracies.append(accuracy)
        
        # Save the trained model for this fold
        model_path = f'models/urban_sound_classifier_fold{test_fold}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model for fold {test_fold} saved to {model_path}")
    
    # Print the average accuracy across all folds
    mean_accuracy = sum(all_accuracies) / len(all_accuracies)
    print("\n---------- RESULTS ----------")
    print(f"Average accuracy across all 10 folds: {mean_accuracy:.2f}%")
    print(f"Individual fold accuracies: {all_accuracies}")
    
    # Save the final results to a text file
    with open('models/final_results.txt', 'w') as f:
        f.write(f"Average accuracy: {mean_accuracy:.2f}%\n")
        for fold, acc in enumerate(all_accuracies, 1):
            f.write(f"Fold {fold}: {acc:.2f}%\n")
    print("Results saved to models/final_results.txt")



if __name__ == "__main__":
    train_urban_sound_classifier()