import os
import cv2
import numpy as np
from glob import glob
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models

# --------------------------
# 1. Dataset Definition
# --------------------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        Args:
          root_dir: Directory containing two subfolders 'sludge' and 'non_sludge'
          num_frames: Number of frames to sample per video.
          transform: Optional torchvision transform for frame preprocessing.
        """
        self.video_paths = []
        self.labels = []
        # Label mapping: non_sludge=0, sludge=1
        for label, subdir in enumerate(['non_sludge', 'sludge']):
            dir_path = os.path.join(root_dir, subdir)
            for file in glob(os.path.join(dir_path, '*.mp4')):
                self.video_paths.append(file)
                self.labels.append(label)
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        # Read video using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Uniformly sample indices (if not enough frames, use all available)
        if total_frames < self.num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        frame_idx = 0
        selected_count = 0
        success = True
        while success and selected_count < len(indices):
            success, frame = cap.read()
            if not success:
                break
            if frame_idx == indices[selected_count]:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 256x256 then center crop to 224x224
                frame = cv2.resize(frame, (256, 256))
                start = (256 - 224) // 2
                frame = frame[start:start+224, start:start+224]
                if self.transform:
                    frame = self.transform(frame)
                else:
                    # Convert to tensor and normalize using ImageNet stats
                    frame = transforms.ToTensor()(frame)
                    frame = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(frame)
                frames.append(frame)
                selected_count += 1
            frame_idx += 1
        cap.release()
        # Pad with zeros if not enough frames
        if len(frames) < self.num_frames:
            pad = [torch.zeros_like(frames[0])] * (self.num_frames - len(frames))
            frames.extend(pad)
        # Shape: (T, C, H, W)
        video_tensor = torch.stack(frames)
        return video_tensor, label

# --------------------------
# 2. Model Definition
# --------------------------
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoClassifier, self).__init__()
        # Use pretrained ResNet50 backbone (remove avgpool and fc)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Output: (B, 2048, H', W')
        self.backbone = nn.Sequential(*modules)
        # Spatial attention module: a 1x1 convolution to compute an attention map
        self.attn_conv = nn.Conv2d(2048, 1, kernel_size=1)
        # Post-aggregation layers
        self.bn = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        frame_feats = []
        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            feat_map = self.backbone(frame)  # (B, 2048, h, w)
            # Compute spatial attention
            attn = torch.sigmoid(self.attn_conv(feat_map))  # (B, 1, h, w)
            weighted_feat = feat_map * attn  # Element-wise multiply
            # Global average pooling over spatial dimensions
            pooled = weighted_feat.view(B, 2048, -1).mean(dim=2)  # (B, 2048)
            frame_feats.append(pooled)
        # Aggregate features over time (average pooling)
        video_feat = torch.stack(frame_feats, dim=1).mean(dim=1)  # (B, 2048)
        video_feat = self.bn(video_feat)
        video_feat = self.dropout(video_feat)
        logits = self.fc(video_feat)  # (B, num_classes)
        return logits

# --------------------------
# 3. Early Stopping Utility
# --------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# --------------------------
# 4. Training, Validation, and Testing Functions
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for videos, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * videos.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                videos = videos.to(device)
                labels = labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * videos.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        # Early stopping check
        early_stopping(val_loss, model, "best_model.pth")
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    # Load best checkpoint
    model.load_state_dict(torch.load("best_model.pth"))
    return model

def test_model(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Testing"):
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

# --------------------------
# 5. Main Training Script
# --------------------------
if __name__ == '__main__':
    # For reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Directory structure: ./dataset/sludge and ./dataset/non_sludge
    dataset_dir = "./dataset/video"
    full_dataset = VideoDataset(dataset_dir, num_frames=16)
    dataset_size = len(full_dataset)
    # Split into 70% train, 15% validation, 15% test
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model = VideoClassifier(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=5, verbose=True)

    num_epochs = 30
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping)

    # Evaluate on test set
    test_acc = test_model(model, test_loader, device)

    # Save the final model
    torch.save(model.state_dict(), "final_video_classifier.pth")
    print("Training complete and model saved.")