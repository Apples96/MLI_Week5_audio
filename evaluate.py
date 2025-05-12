import os
import torch
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from model import UrbanSoundClassifier
from data import UrbanSoundDataset
from datetime import datetime

def load_model(model_path, max_seq_length=1501):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the saved model file
        max_seq_length: Maximum sequence length the model was trained with
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        The loaded model
    """
    # Initialize model with the same parameters used during training
    model = UrbanSoundClassifier(max_seq_length=max_seq_length)
    
    # Load saved parameters
    model.load_state_dict(torch.load(model_path))
    
    # Move model to appropriate device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.to(device)
    model.eval()
    
    return model, device

def evaluate_random_examples(model_path, pickle_path="./data/processed_urbansound8k.pkl", 
                            test_fold=1, num_examples=5, max_seq_length=1501):
    """
    Evaluate a trained model on random examples from the test set
    
    Args:
        model_path: Path to the saved model file
        pickle_path: Path to the processed dataset
        test_fold: Which fold to use for testing
        num_examples: Number of random examples to evaluate
        max_seq_length: Maximum sequence length
    """
    # Load model
    model, device = load_model(model_path, max_seq_length)
    
    # Load dataset
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Create dataset for the test fold
    full_dataset = UrbanSoundDataset(dataset, max_length=max_seq_length)
    test_indices = [i for i in range(len(full_dataset)) if full_dataset[i]['fold'] == test_fold]
    
    # Randomly select examples
    selected_indices = random.sample(test_indices, min(num_examples, len(test_indices)))
    
    print(f"\nEvaluating {num_examples} random examples from fold {test_fold}:")
    print("-" * 60)
    
    class_names = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
    ]
    
    correct = 0
    
    for idx in selected_indices:
        sample = full_dataset[idx]
        spectro = sample['log_mel_spectrogram'].unsqueeze(0).to(device)  # Add batch dimension
        target = sample['classID'].item()
        
        # Make prediction
        with torch.no_grad():
            logits = model(spectro)
        
        # Get top prediction
        probabilities = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0, pred_class].item() * 100
        
        # Format result
        result = "CORRECT" if pred_class == target else "WRONG"
        if pred_class == target:
            correct += 1
            
        print(f"Example {idx} (from fold {test_fold}):")
        print(f"  True class: {class_names[target]} (ID: {target})")
        print(f"  Predicted: {class_names[pred_class]} (ID: {pred_class}) with {confidence:.1f}% confidence")
        print(f"  Result: {result}")
        print("-" * 60)
    
    print(f"Accuracy on random samples: {correct}/{num_examples} ({100 * correct / num_examples:.1f}%)")
    return correct / num_examples

def run_fold_evaluation(model_path, pickle_path="./data/processed_urbansound8k.pkl", 
                       test_fold=1, batch_size=32, max_seq_length=1501, 
                       save_results=True, plot_cm=True):
    """
    Run a complete evaluation on a specific fold
    
    Args:
        model_path: Path to the saved model file
        pickle_path: Path to the processed dataset
        test_fold: Which fold to use for testing
        batch_size: Batch size for evaluation
        max_seq_length: Maximum sequence length
        save_results: Whether to save results to disk
        plot_cm: Whether to plot and save the confusion matrix
        
    Returns:
        accuracy: The accuracy on the test fold
    """
    # Load model
    model, device = load_model(model_path, max_seq_length)
    
    # Load dataset
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Create dataset and dataloader for the test fold
    full_dataset = UrbanSoundDataset(dataset, max_length=max_seq_length)
    test_indices = [i for i in range(len(full_dataset)) if full_dataset[i]['fold'] == test_fold]
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Prepare for evaluation
    all_preds = []
    all_targets = []
    
    # Evaluate
    print(f"\nEvaluating model on fold {test_fold}...")
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            # Move data to device
            log_mel_spectrogram = batch['log_mel_spectrogram'].to(device)
            targets = batch['classID'].to(device)
            
            # Forward pass
            logits = model(log_mel_spectrogram)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    class_names = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
    ]
    
    # Create class report
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names,
        digits=3
    )
    
    # Print results
    print(f"\nResults for fold {test_fold}:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"fold{test_fold}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        # Plot and save confusion matrix if requested
        if plot_cm:
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Fold {test_fold}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
            print(f"Saved confusion matrix to {os.path.join(results_dir, 'confusion_matrix.png')}")
    
    return accuracy

def run_all_folds_evaluation(models_dir, pickle_path="./data/processed_urbansound8k.pkl", 
                            batch_size=32, max_seq_length=1501):
    """
    Evaluate models for all folds and compute the average accuracy
    
    Args:
        models_dir: Directory containing saved models for each fold
        pickle_path: Path to the processed dataset
        batch_size: Batch size for evaluation
        max_seq_length: Maximum sequence length
    """
    print("\nRunning 10-fold cross-validation evaluation...")
    
    # Find models for each fold
    fold_accuracies = []
    fold_model_paths = {}
    
    # First, identify the models for each fold
    for filename in os.listdir(models_dir):
        if filename.endswith('.pt'):
            for fold in range(1, 11):
                if f"fold{fold}_" in filename:
                    fold_model_paths[fold] = os.path.join(models_dir, filename)
                    break
    
    # Ensure we have models for all folds
    missing_folds = [fold for fold in range(1, 11) if fold not in fold_model_paths]
    if missing_folds:
        print(f"Warning: Missing models for folds {missing_folds}")
        print("Will only evaluate available folds.")
    
    print(f"Found models for folds: {sorted(fold_model_paths.keys())}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"all_folds_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate each fold
    all_fold_results = {}
    for fold, model_path in sorted(fold_model_paths.items()):
        print(f"\n{'-' * 40}")
        print(f"Evaluating fold {fold} with model: {os.path.basename(model_path)}")
        accuracy = run_fold_evaluation(
            model_path=model_path,
            pickle_path=pickle_path,
            test_fold=fold,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            save_results=True,
            plot_cm=True
        )
        fold_accuracies.append(accuracy)
        all_fold_results[fold] = accuracy * 100
    
    # Calculate and save final results
    mean_accuracy = np.mean(fold_accuracies) * 100
    std_accuracy = np.std([acc * 100 for acc in fold_accuracies])
    
    print("\n" + "=" * 60)
    print(f"10-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Mean accuracy: {mean_accuracy:.2f}% (±{std_accuracy:.2f}%)")
    for fold, acc in all_fold_results.items():
        print(f"Fold {fold}: {acc:.2f}%")
    
    # Plot fold accuracies
    plt.figure(figsize=(10, 6))
    folds = list(all_fold_results.keys())
    accuracies = [all_fold_results[f] for f in folds]
    plt.bar(folds, accuracies)
    plt.axhline(y=mean_accuracy, color='r', linestyle='-', label=f'Mean: {mean_accuracy:.2f}%')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Fold - 10-Fold Cross-Validation')
    plt.xticks(folds)
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "fold_accuracies.png"))
    
    # Save final results
    with open(os.path.join(results_dir, "final_results.txt"), "w") as f:
        f.write(f"10-FOLD CROSS-VALIDATION RESULTS\n")
        f.write(f"Mean accuracy: {mean_accuracy:.2f}% (±{std_accuracy:.2f}%)\n\n")
        for fold, acc in all_fold_results.items():
            f.write(f"Fold {fold}: {acc:.2f}%\n")
    
    print(f"\nResults saved to {results_dir}")
    return mean_accuracy, std_accuracy, all_fold_results

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # You can call the functions individually:
    
    # Option 1: Evaluate random examples from a specific fold
    model_path = "models/urban_sound_classifier_fold1.pt"
    evaluate_random_examples(model_path, test_fold=10, num_examples=5)
    
    # Option 2: Run full evaluation on a specific fold
    model_path = "models/urban_sound_classifier_fold1.pt"
    run_fold_evaluation(model_path, test_fold=10)
    
    # Option 3: Run 10-fold cross-validation evaluation (recommended)
    # This assumes you have trained models for all 10 folds in the "models" directory
    # run_all_folds_evaluation(models_dir="models")