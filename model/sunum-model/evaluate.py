import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from data_loader import process_pipeline
from model import GuardianHybrid

def evaluate_dataset(model, device, data_path, scaler_path, dataset_type='ids2017', batch_size=32, limit=None):
    print(f"\nEvaluating on {dataset_type.upper()} dataset at {data_path}...")
    
    # Process data
    try:
        seqs, labels, loader = process_pipeline(
            primary_path=data_path,
            scaler_path=scaler_path,
            mode='train_classifier', # We need labels to evaluate
            dataset_type=dataset_type,
            limit=limit
        )
    except Exception as e:
        print(f"Error processing data for {dataset_type}: {e}")
        return

    if seqs.size == 0 or labels.size == 0:
        print(f"No valid data found or processed for {dataset_type}.")
        return

    print(f"Data processed successfully. Sequences shape: {seqs.shape}")

    dataset = TensorDataset(torch.Tensor(seqs), torch.Tensor(labels).long())
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            # Forward pass in classify mode
            probs = model(data, mode='classify')
            _, predicted = torch.max(probs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Results for {dataset_type.upper()}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    target_names = ['BENIGN', 'DDoS', 'PortScan', 'WebAttack', 'Bot']
    # Filter target names based on unique labels in the dataset
    unique_labels = sorted(list(set(all_labels)))
    present_target_names = [target_names[i] for i in unique_labels]
    
    print(classification_report(all_labels, all_preds, target_names=present_target_names, labels=unique_labels, zero_division=0))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Guardian Model on CIC-IDS Datasets")
    parser.add_argument('--limit', type=int, default=None, help="Limit rows for debugging/faster evaluation")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models_dir = os.path.join(base_dir, 'backend', 'saved_models')
    model_path = os.path.join(models_dir, 'guardian_complete.pth')
    scaler_path = os.path.join(models_dir, 'guardian_scaler.pkl')
    
    data_dir = os.path.join(base_dir, 'backend', 'datasets', 'raw')
    path_2017 = os.path.join(data_dir, 'CIC-IDS 2017')
    path_2018 = os.path.join(data_dir, 'CIC-IDS 2018')

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"Scaler file not found: {scaler_path}")
        return

    # To initialize the model, we need the input dimension.
    # The processed sequences usually have 69 or 70 features.
    # Let's peek at the scaler's feature count if possible, or we can hardcode 70 and rely on the model definition.
    # The best way is to load scaler and check `n_features_in_`.
    import joblib
    try:
        scaler = joblib.load(scaler_path)
        input_dim = scaler.n_features_in_
        print(f"Loaded scaler with input dimension: {input_dim}")
    except Exception as e:
        print(f"Could not determine input dimension from scaler, assuming 70. Error: {e}")
        input_dim = 70

    print("Initializing GuardianHybrid Model...")
    model = GuardianHybrid(input_dim=input_dim).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # Evaluate on 2017
    if os.path.exists(path_2017):
        evaluate_dataset(model, device, path_2017, scaler_path, dataset_type='ids2017', batch_size=args.batch_size, limit=args.limit)
    else:
        print(f"Dataset 2017 not found at {path_2017}")

    # Evaluate on 2018
    if os.path.exists(path_2018):
        evaluate_dataset(model, device, path_2018, scaler_path, dataset_type='ids2018', batch_size=args.batch_size, limit=args.limit)
    else:
        print(f"Dataset 2018 not found at {path_2018}")

if __name__ == "__main__":
    main()
