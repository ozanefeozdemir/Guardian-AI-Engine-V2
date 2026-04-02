
import os
import sys
import numpy as np
import pandas as pd

# Try importing torch, handle if missing (since user handles env)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not found. Please install it (with CUDA support if needed) to run the training loop.")
    TORCH_AVAILABLE = False

from data_loader import process_pipeline, GuardianDataLoader
from model import GuardianHybrid

def main():
    print("=== Guardian NIDS: Execution Script ===")
    
    # Configuration
    # Using a small seq_len for demo
    SEQ_LEN = 10
    BATCH_SIZE = 32
    EPOCHS = 1 # Demo only
    LR = 0.001
    
    # Path Setup
    # Adjust these paths as needed based on where you run the script
    # Current: engine/main.py -> Base: guardian-engine/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    primary_data_path = os.path.join(base_dir, 'data', 'train', 'ids-2017')
    secondary_data_path = os.path.join(base_dir, 'data', 'train', 'UNSW') # Example placeholder
    scaler_path = os.path.join(base_dir, 'checkpoints', 'guardian_scaler.pkl')

    # Step 1: Data Processing
    # Check if data exists, otherwise generate dummy data for demo
    if os.path.exists(primary_data_path) and os.listdir(primary_data_path):
        print(f"Data found at {primary_data_path}. Processing...")
        # Note: process_pipeline might take a while on full dataset. 
        # For this script we might want to limit it or just run it.
        try:
            # We can't easily limit rows inside process_pipeline without modifying it 
            # or passing args. For now, we trust the user has resources or will modify.
            sequences, _, loader = process_pipeline(primary_data_path, secondary_data_path, scaler_path, mode='train_autoencoder')
            
            # Use only a subset for quick demo if it's huge
            if len(sequences) > 1000:
                print(f"Using 1000 samples for demo out of {len(sequences)}")
                sequences = sequences[:1000]
                
        except Exception as e:
            print(f"Error in data pipeline: {e}")
            return
    else:
        print("Data directory not found or empty. Generating DUMMY data for demonstration.")
        # Dummy data: (100 samples, 10, 70 features) assuming ~70 features after cleaning
        n_features = 70
        sequences = np.random.rand(100, SEQ_LEN, n_features).astype(np.float32)
        print(f"Generated dummy sequences: {sequences.shape}")

    if sequences.size == 0:
        print("No sequences generated. Exiting.")
        return

    n_features = sequences.shape[2]
    
    # Step 2: Model Initialization
    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Integration stops here.")
        return

    print("Initializing GuardianHybrid Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GuardianHybrid(input_dim=n_features, seq_len=SEQ_LEN, latent_dim=32, n_classes=5).to(device)
    print(model)

    # Step 3: Dummy Training Loop (Autoencoder Phase)
    # We train on BENIGN data only (reconstruction)
    print("\nStarting Autoencoder Training (MSE Loss on Benign)...")
    
    # Verify data type
    tensor_x = torch.Tensor(sequences) # (N, 10, 70)
    
    # Create DataLoader
    dataset = TensorDataset(tensor_x, tensor_x) # Target is input for Autoencoder
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass in autoencoder mode
            output = model(data, mode='train_autoencoder')
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.6f}")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.6f}")

    print("\nTraining Demo Complete.")
    print("To train the Classifier, you would freeze the encoder/LSTM and use 'classify' mode with labeled attack data.")

if __name__ == "__main__":
    main()
