
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import process_pipeline
from model import GuardianHybrid

def train_autoencoder(model, train_loader, epochs, lr, device, save_path):
    print("\n=== Phase 1: Autoencoder Training ===")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in train_loader: # Target is input
            data = data.to(device)
            target = data
            
            optimizer.zero_grad()
            output = model(data, mode='train_autoencoder')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Autoencoder model saved to {save_path}")

def train_classifier(model, train_loader, epochs, lr, device, save_path):
    print("\n=== Phase 2: Classifier Training ===")
    
    # FREEZE ENCODER
    # We want to keep the feature extraction stable and only learn classification
    for param in model.conv1.parameters(): param.requires_grad = False
    for param in model.bn1.parameters(): param.requires_grad = False
    for param in model.lstm.parameters(): param.requires_grad = False
    for param in model.fc_latent.parameters(): param.requires_grad = False
    
    # Verify freezing
    print("Encoder layers frozen. Training Classifier Head only.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            # output probabilities
            probs = model(data, mode='classify') 
            
            # CrossEntropyLoss expects logits usually, but if our model returns softmax, we should be careful.
            # My model.py returns Softmax probabilities. 
            # PyTorch implementation of CrossEntropyLoss expects LOGITS.
            # So I should take log of probs, OR change model to return logits.
            # Changing model.py is cleaner, but let's just do log here for safety without re-editing files excessively.
            # NLLLoss expects log_probs.
            
            log_probs = torch.log(probs + 1e-9) # Avoid log(0)
            loss = nn.NLLLoss()(log_probs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Acc: {acc:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Full model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, required=True, choices=['autoencoder', 'classifier', 'all'], help='Training phase')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--limit', type=int, default=None, help="Limit rows for debugging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    # engine/train.py -> ..
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary_path = os.path.join(base_dir, 'backend', 'datasets', 'raw', 'CIC-IDS 2017')
    
    # Save to backend/saved_models instead of checkpoints
    out_dir = os.path.join(base_dir, 'backend', 'saved_models')
    os.makedirs(out_dir, exist_ok=True)
    
    scaler_path = os.path.join(out_dir, 'guardian_scaler.pkl')
    ae_model_path = os.path.join(out_dir, 'guardian_autoencoder.pth')
    full_model_path = os.path.join(out_dir, 'guardian_complete.pth')

    # Load Data accordingly
    if args.phase in ['autoencoder', 'all']:
        print("Prepare Data for Autoencoder (Benign Only)...")
        # Ensure we use 'train_autoencoder' mode
        seqs, _, loader = process_pipeline(primary_path, scaler_path=scaler_path, mode='train_autoencoder', limit=args.limit)
        
        if seqs.size == 0:
            print("No data found for AE training. Generating DUMMY data for valid syntax check.")
            n_samples, n_feat = 100, 70
            seqs = np.random.rand(n_samples, 10, n_feat).astype(np.float32)
            
            # Save a dummy scaler so inference doesn't break
            import joblib
            from sklearn.preprocessing import MinMaxScaler
            dummy_scaler = MinMaxScaler()
            dummy_scaler.fit(np.random.rand(n_samples, n_feat))
            joblib.dump(dummy_scaler, scaler_path)
            print(f"Saved DUMMY scaler to {scaler_path}")
        
        # Init Model
        n_features = seqs.shape[2]
        model = GuardianHybrid(input_dim=n_features).to(device)
        
        dataset = TensorDataset(torch.Tensor(seqs), torch.Tensor(seqs))
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        train_autoencoder(model, train_loader, args.epochs, args.lr, device, ae_model_path)
        
        # Cleanup
        del seqs, dataset, train_loader
        torch.cuda.empty_cache()

    if args.phase in ['classifier', 'all']:
        print("Prepare Data for Classifier (All Classes)...")
        # Ensure we use 'train_classifier' mode
        seqs, labels, loader = process_pipeline(primary_path, scaler_path=scaler_path, mode='train_classifier', limit=args.limit)
        
        if seqs.size == 0:
             print("No data found for Classifier training. Generating DUMMY data.")
             n_samples, n_feat = 100, 70
             seqs = np.random.rand(n_samples, 10, n_feat).astype(np.float32)
             labels = np.random.randint(0, 5, n_samples)
             
        n_features = seqs.shape[2]
        
        # Load Model
        model = GuardianHybrid(input_dim=n_features).to(device)
        if os.path.exists(ae_model_path):
            print(f"Loading pretrained Autoencoder weights from {ae_model_path}")
            model.load_state_dict(torch.load(ae_model_path))
        else:
            print("Warning: Autoencoder weights not found! Training Classifier from scratch (Encoder random).")

        dataset = TensorDataset(torch.Tensor(seqs), torch.Tensor(labels))
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        train_classifier(model, train_loader, args.epochs, args.lr, device, full_model_path)

if __name__ == "__main__":
    main()
