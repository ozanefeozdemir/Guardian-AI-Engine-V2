
import torch
import torch.nn as nn
import torch.nn.functional as F

class GuardianHybrid(nn.Module):
    def __init__(self, input_dim, seq_len=10, latent_dim=32, n_classes=5):
        super(GuardianHybrid, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # --- Hybrid Encoder (Stage 1) ---
        # 1. Spatial Feature Extraction (Conv1d)
        # Input: (Batch, Seq_Len, Features) -> Permute to (Batch, Features, Seq_Len) for Conv1d
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # 2. Temporal Feature Extraction (LSTM)
        # Random choice: 128 hidden units
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        
        # Bottleneck to Latent Vector
        self.fc_latent = nn.Linear(128, latent_dim)

        # --- Decoder (Reconstruction) ---
        # We need to reconstruction (Batch, Seq_Len, Input_Dim)
        # Simple approach: Latent -> Hidden -> Seq_Len * Input_Dim
        self.decoder_fc = nn.Linear(latent_dim, 128)
        self.decoder_out = nn.Linear(128, seq_len * input_dim)

        # --- Classifier Head (Stage 2) ---
        # Classes: Benign, DDoS, PortScan, WebAttack, Botnet (Total 5)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, mode='train_autoencoder'):
        """
        Forward pass with mode switching.
        x shape: (Batch, Seq_Len, Features)
        """
        batch_size = x.size(0)

        # --- Encoder Pass ---
        # Conv1d expects (Batch, Channels, Length)
        x_perm = x.permute(0, 2, 1) # (Batch, Features, Seq_Len)
        
        c_out = F.relu(self.bn1(self.conv1(x_perm)))
        
        # LSTM expects (Batch, Length, Features) -> Permute back
        lstm_in = c_out.permute(0, 2, 1) # (Batch, Seq_Len, 64)
        
        _, (h_n, _) = self.lstm(lstm_in)
        # h_n shape: (1, Batch, 128) -> Squeeze to (Batch, 128)
        lstm_feat = h_n[-1]
        
        latent_vector = self.fc_latent(lstm_feat) # (Batch, Latent_Dim)

        # --- Mode Switching ---
        if mode == 'train_autoencoder':
            # Reconstruction
            dec_hidden = F.relu(self.decoder_fc(latent_vector))
            reconstruction = self.decoder_out(dec_hidden)
            # Reshape back to (Batch, Seq_Len, Features)
            reconstruction = reconstruction.view(batch_size, self.seq_len, self.input_dim)
            return reconstruction
            
        elif mode == 'classify':
            # Classification
            logits = self.classifier(latent_vector)
            # We usually return logits for CrossEntropyLoss, or Softmax for inference.
            probs = F.softmax(logits, dim=1)
            return probs
            
        else:
            # Maybe return both? or latent?
            return latent_vector
