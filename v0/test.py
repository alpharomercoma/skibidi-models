import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models


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

# Reuse the crop_center_square and load_video functions from training
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=16, resize=(256, 256)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            # Convert from BGR to RGB
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Define the same preprocessing as used during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def prepare_video_for_inference(video_path, num_frames=16, img_size=224):
    # Load frames (resize to 256x256 first)
    frames = load_video(video_path, max_frames=num_frames, resize=(256, 256))
    if frames.size == 0:
        raise ValueError(f"No frames extracted from video {video_path}. Check the file path or file integrity.")

    processed_frames = []
    # Center crop to 224x224 and apply transforms
    start = (256 - img_size) // 2
    for frame in frames:
        frame = frame[start:start+img_size, start:start+img_size]
        frame = transform(frame)
        processed_frames.append(frame)

    # If no frames were processed, raise an error
    if len(processed_frames) == 0:
        raise ValueError(f"No frames processed from video {video_path}.")

    # If fewer than num_frames, pad with zero tensors based on the first frame's shape
    if len(processed_frames) < num_frames:
        pad = [torch.zeros_like(processed_frames[0])] * (num_frames - len(processed_frames))
        processed_frames.extend(pad)

    video_tensor = torch.stack(processed_frames)  # Shape: (T, C, H, W)
    return video_tensor


def classify_video(video_path, model, device, num_frames=16):
    # Prepare the video tensor
    video_tensor = prepare_video_for_inference(video_path, num_frames=num_frames, img_size=224)
    # Add batch dimension: (1, T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(video_tensor)  # (1, num_classes)
        probabilities = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probabilities, dim=1)

    class_names = ['non_sludge', 'sludge']
    predicted_class = class_names[pred.item()]
    return predicted_class, confidence.item()

# Example usage:
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model (same architecture as used for training)
    model = VideoClassifier(num_classes=2)
    # Load the final trained model
    model.load_state_dict(torch.load("./final_video_classifier.pth", map_location=device))

    # Path to an input video provided by the user
    input_video_path = "./manual/sludge_2.mp4"
    pred_class, conf = classify_video(input_video_path, model, device, num_frames=16)
    print(f"Prediction: {pred_class} with confidence {conf*100:.2f}%")
