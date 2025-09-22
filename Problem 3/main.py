import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.signal import butter, filtfilt

# -------------------------
# Config (updated)
# -------------------------
PROFILING_NPZ = "datasetB.npz"
TARGET_NPZ = "datasetA.npz"
USE_HW_LABEL = True  # Use Hamming Weight labels (easier to learn)
BATCH_SIZE = 128  # Adjust based on GPU memory; reduce if memory error
NUM_EPOCHS = 100  # More epochs for better training
LR = 3e-4  # Lower learning rate for stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-40

# Full AES S-box (256 elements)
SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)

def hamming_weight_array(x):
    return np.unpackbits(x[:, None], axis=1).sum(axis=1)

def load_npz(path, has_key=True):
    data = np.load(path)
    traces = data["trace"].astype(np.float32)
    plains = data["plaintext"].astype(np.uint8)
    keys = data["key"].astype(np.uint8) if has_key else None
    return plains, keys, traces

def make_labels(plains, keys, use_hw=False):
    xor = np.bitwise_xor(plains, keys)
    s = SBOX[xor]
    return hamming_weight_array(s) if use_hw else s

class Enhanced1DCNN(nn.Module):
    def __init__(self, n_classes=256):
        super().__init__()
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128, kernel_size=9)
        self.res_block2 = self._make_residual_block(128, 256, kernel_size=7)
        self.res_block3 = self._make_residual_block(256, 512, kernel_size=5)
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def _make_residual_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        # Residual blocks with downsampling
        x = self.res_block1(x)
        x = nn.MaxPool1d(2)(x)  # Downsample
        x = self.res_block2(x)
        x = nn.MaxPool1d(2)(x)  # Downsample
        x = self.res_block3(x)
        x = nn.MaxPool1d(2)(x)  # Downsample
        # Apply attention
        att = self.attention(x)
        x = x * att
        # Global pooling and classification
        x = self.global_pool(x)
        return self.classifier(x)

class TracesDataset(Dataset):
    def __init__(self, traces, labels=None, augment=False):
        self.traces = torch.from_numpy(traces).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.augment = augment

    def __len__(self):
        return self.traces.shape[0]

    def __getitem__(self, idx):
        x = self.traces[idx].unsqueeze(0)  # (1,L)
        # Apply augmentation during training
        if self.augment and self.labels is not None:
            # Add Gaussian noise
            if torch.rand(1) > 0.5:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            # Random scaling
            if torch.rand(1) > 0.5:
                scale = 0.9 + 0.2 * torch.rand(1)
                x = x * scale
        return (x, self.labels[idx]) if self.labels is not None else x

def train_model(model, train_loader, val_loader):
    # Use label smoothing for better generalization
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW optimizer with weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    best = None
    best_loss = 1e9
    patience = 15
    patience_counter = 0

    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        tot = 0
        correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()

        # Update learning rate
        scheduler.step()
        tloss = tot / len(train_loader.dataset)
        tacc = correct / len(train_loader.dataset)

        # Validation
        model.eval()
        vloss = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = ce(logits, yb)
                vloss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()

        vloss /= len(val_loader.dataset)
        vacc = correct / len(val_loader.dataset)
        print(f"Epoch {ep}: train_loss={tloss:.4f}, train_acc={tacc:.3f}, "
              f"val_loss={vloss:.4f}, val_acc={vacc:.3f}, lr={opt.param_groups[0]['lr']:.6f}")

        # Early stopping with patience
        if vloss < best_loss:
            best_loss = vloss
            best = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    model.load_state_dict(best)
    return model

def predict_probs(model, loader):
    model.eval()
    allp = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(DEVICE)
            p = torch.softmax(model(xb), dim=1).cpu().numpy()
            allp.append(p)
    return np.vstack(allp)

def score_keys(probs, plains, use_hw=False):
    N, C = probs.shape
    scores = np.zeros(256, dtype=np.float64)
    for k in range(256):
        z = SBOX[np.bitwise_xor(plains, k)]
        if use_hw:
            z = hamming_weight_array(z)
        probz = probs[np.arange(N), z]
        probz = np.maximum(probz, EPS)
        scores[k] = np.sum(np.log(probz))
    return scores

def preprocess_traces(traces):
    """Apply preprocessing to enhance signal features"""
    # 1. Standardize
    mean = traces.mean(axis=0, keepdims=True)
    std = traces.std(axis=0, keepdims=True) + 1e-9
    traces = (traces - mean) / std

    # 2. Apply bandpass filter to remove noise
    def bandpass_filter(data, low=0.1, high=0.4, fs=1.0, order=3):
        nyq = 0.5 * fs
        low_cut = low / nyq
        high_cut = high / nyq
        b, a = butter(order, [low_cut, high_cut], btype='band')
        return filtfilt(b, a, data)

    # Apply filter to each trace
    filtered_traces = np.zeros_like(traces)
    for i in range(traces.shape[0]):
        filtered_traces[i] = bandpass_filter(traces[i])

    # 3. Compute derivative to highlight transitions
    derivative = np.diff(filtered_traces, axis=1)
    derivative = np.pad(derivative, ((0, 0), (0, 1)), mode='edge')

    # 4. Combine original and derivative
    combined = np.concatenate([filtered_traces, derivative], axis=1)

    return combined

def main():
    # Load profiling data
    plains_B, keys_B, traces_B = load_npz(PROFILING_NPZ, has_key=True)
    labels_B = make_labels(plains_B, keys_B, USE_HW_LABEL)
    n_classes = 9 if USE_HW_LABEL else 256

    # Preprocess traces
    traces_B = preprocess_traces(traces_B)

    # Split data
    Xtr, Xval, ytr, yval = train_test_split(
        traces_B, labels_B, test_size=0.15, random_state=42, stratify=labels_B
    )

    # Create datasets with augmentation for training
    tr_loader = DataLoader(
        TracesDataset(Xtr, ytr, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Changed to 0 to avoid multiprocessing issues on Windows
    )
    val_loader = DataLoader(
        TracesDataset(Xval, yval),
        batch_size=BATCH_SIZE,
        num_workers=0  # Changed to 0 to avoid multiprocessing issues on Windows
    )

    # Train model
    model = Enhanced1DCNN(n_classes=n_classes).to(DEVICE)
    model = train_model(model, tr_loader, val_loader)

    # Load target data
    # Note: Challenge states datasetA has no keys, so has_key=False
    plains_A, keys_A, traces_A = load_npz(TARGET_NPZ, has_key=False)
    traces_A = preprocess_traces(traces_A)

    # Use the same preprocessing parameters as training data
    mean = traces_B.mean(axis=0, keepdims=True)
    std = traces_B.std(axis=0, keepdims=True) + 1e-9
    traces_A = (traces_A - mean) / std

    target_loader = DataLoader(
        TracesDataset(traces_A),
        batch_size=BATCH_SIZE,
        num_workers=0  # Changed to 0 to avoid multiprocessing issues on Windows
    )
    probs_A = predict_probs(model, target_loader)

    # Score keys
    scores = score_keys(probs_A, plains_A, USE_HW_LABEL)
    ranked = np.argsort(-scores)

    # Verification (manual input since keys_A is None)
    print("\nTop 10 key guesses:")
    for i, k in enumerate(ranked[:10]):
        print(f"#{i+1} Key={k:02X}, Score={scores[k]:.2f}")

    # Manual verification (input true key from organizers if known)
    true_key_byte = input("Enter true key byte (hex, e.g., 0xFA) or press Enter to skip: ").strip()
    if true_key_byte:
        try:
            true_k = int(true_key_byte, 16)
            rank = np.where(ranked == true_k)[0][0] + 1  # 1-based rank
            print(f"\nVerification Results:")
            print(f"Actual first key byte: {true_k:02X}")
            print(f"Top guess: {ranked[0]:02X}")
            print(f"Correct key rank: {rank}")
            print(f"Attack success: {'YES' if rank == 1 else 'NO'}")
        except ValueError:
            print("Invalid hex input—skipping verification.")
    else:
        print("\nVerification Results: Provide true key byte to organizers for confirmation.")

if __name__ == "__main__":
    main()
