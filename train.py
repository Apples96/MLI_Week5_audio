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
import wandb



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




def train_urban_sound_classifier(test_fold = 10, config=None):
    # Initialize wandb with configuration
    if config is None:
        config = {
            "learning_rate": 0.0001,
            "epochs": 3,
            "batch_size": 32,
            "model_type": "UrbanSoundClassifier",
            "test_fold": test_fold,
            "max_seq_length": 1502,
            "embed_dim": 384,
            "num_heads": 6,
            "num_layers": 6,
            "optimizer": "Adam"
        }

    # Initialize wandb run
    run = wandb.init(
        project="urban-sound-classification",
        config=config,
        name=f"fold_{test_fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job_type="training"
    )
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Perform 10-fold cross validation
    # all_accuracies = []
    
    # for test_fold in range(1, 11):  # 10-fold cross validation
    # print(f"\n---------- FOLD {test_fold} ----------") # 10-fold cross validation
        
    # Get dataloaders for this fold
    train_dataset, test_dataset, train_dataloader, test_dataloader = get_data_loader(test_fold=test_fold)
    
    # Initialize model, loss function, and optimizer
    
    model = UrbanSoundClassifier(
        max_seq_length=config["max_seq_length"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"]
    )
    # Log model architecture to wandb
    wandb.watch(model, log="all", log_freq=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    wandb.log({"device": str(device)})
    model.to(device)

    # Training
    for epoch in range(config["epochs"]):
        model.train()
        epoch_train_loss = 0
        train_correct = 0
        
        for idx, batch in enumerate(train_dataloader):
            # Move data to device
            log_mel_spectrogram = batch['log_mel_spectrogram'].to(device)
            classID = batch['classID'].to(device)
            
            optimizer.zero_grad()
            logits = model(log_mel_spectrogram)  
            loss = criterion(logits, classID)
            epoch_train_loss += loss.item()
            
            # Get predictions for accuracy calculation
            pred = logits.argmax(dim=1)
            train_correct += (pred == classID).sum().item()

            loss.backward()
            optimizer.step()

            # Print stats 
            if idx % 10 == 0:  # Show update every 10 batches
                print(f'Fold {test_fold}, Epoch {epoch} [{idx * len(log_mel_spectrogram)}/{len(train_dataset)} '
                        f'({100. * idx * len(log_mel_spectrogram) / len(train_dataset):.0f}%)]\tLoss: {loss.item():.6f}')
                
                # Log batch statistics to wandb
                wandb.log({
                    "batch": idx + epoch * len(train_dataloader),
                    "batch_loss": loss.item(),
                    "batch_progress": 100. * idx * len(log_mel_spectrogram) / len(train_dataset)
                })
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_accuracy = 100. * train_correct / len(train_dataset)

        model_path = f'models/urban_sound_classifier_fold{test_fold}.pt'
        torch.save(model.state_dict(), model_path)

        # Test model after each epoch
        model.eval()
        epoch_test_loss = 0
        test_correct = 0
        all_preds = []
        all_labels = []

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
                test_correct += (pred == classID).sum().item()


        # Calculate test metrics
        avg_test_loss = epoch_test_loss / len(test_dataloader)
        test_accuracy = 100. * test_correct / len(test_dataset)
    
        print(f'Fold {test_fold}, Epoch {epoch}: '
                f'Train loss: {avg_train_loss:.4f}, '
                f'Test loss: {avg_test_loss:.4f}, '
                f'Accuracy: {test_correct}/{len(test_dataset)} '
                f'({test_accuracy:.2f}%)')
        
        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy
        })
        
        # Save the trained model for this fold. Gets replaced every epoch. 
        model_path = f'models/urban_sound_bigclassifier_fold{test_fold}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model for fold {test_fold} saved to {model_path}")

    # Finish the wandb run
    wandb.finish()
    
    # Save the final fold results # 10-fold cross validation
    # all_accuracies.append(accuracy) # 10-fold cross validation
    
    # Print the average accuracy across all folds # 10-fold cross validation
    # mean_accuracy = sum(all_accuracies) / len(all_accuracies) # 10-fold cross validation
    # print("\n---------- RESULTS ----------") # 10-fold cross validation
    # print(f"Average accuracy across all 10 folds: {mean_accuracy:.2f}%") # 10-fold cross validation
    # print(f"Individual fold accuracies: {all_accuracies}") # 10-fold cross validation
    
    # Save the final results to a text file # 10-fold cross validation
    # with open('models/final_results.txt', 'w') as f: # 10-fold cross validation
    #     f.write(f"Average accuracy: {mean_accuracy:.2f}%\n") # 10-fold cross validation
    #     for fold, acc in enumerate(all_accuracies, 1): # 10-fold cross validation
    #         f.write(f"Fold {fold}: {acc:.2f}%\n") # 10-fold cross validation
    # print("Results saved to models/final_results.txt") # 10-fold cross validation




def run_hyperparameter_sweep():
    """
    Run a hyperparameter sweep with wandb
    """
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization for efficient parameter search
        'metric': {
            'name': 'test_accuracy',
            'goal': 'maximize'   
        },
        'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        's': 2
        },
        'parameters': {
            'learning_rate': {
                'min': 0.00001,
                'max': 0.001,
                'distribution': 'log_uniform'
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'num_layers': {
                'values': [4, 6, 8]
            },
            'num_heads': {
                'values': [4, 6, 8]
            },
            'embed_dim': {
                'values': [256, 384, 512]
            }
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="urban-sound-classification")
    
    # Define the training function for the sweep
    def train_sweep():
        # Initialize a new wandb run
        wandb.init()
        
        # Access the hyperparameters
        config = wandb.config
        
        # Train with the current hyperparameters
        train_urban_sound_classifier(test_fold=10, config=config)
    
    # Run the sweep
    wandb.agent(sweep_id, train_sweep, count=10)  # Run 10 experiments


if __name__ == "__main__":
    # For normal training with default parameters
    train_urban_sound_classifier()
    
    # For training while testing a hyperparameter sweep
    # run_hyperparameter_sweep()