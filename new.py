
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD, AdamW 
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import ColorJitter, RandomAffine, RandomPerspective

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        # If window size is larger than input resolution, we don't partition windows
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
            
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x):
        H, W = self.input_resolution
        B, C, h, w = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H*W, C)
        
        shortcut = x
        x = self.norm1(x)
        
        # Reshape for window attention
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            
        x = x.view(B, H*W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back to original format
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    H_pad, W_pad = x.shape[1], x.shape[2]
    
    x = x.view(B, H_pad // window_size, window_size,
               W_pad // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=(16, 16)):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.LayerNorm(out_channels)
        
        # Adjust window size if input is too small
        window_size = min(7, min(input_resolution))
        
        # Create Swin Transformer blocks with shifted and non-shifted windows
        self.block1 = SwinTransformerBlock(
            dim=out_channels,
            input_resolution=input_resolution,
            num_heads=8,
            window_size=window_size,
            shift_size=0
        )
        self.block2 = SwinTransformerBlock(
            dim=out_channels,
            input_resolution=input_resolution,
            num_heads=8,
            window_size=window_size,
            shift_size=window_size//2
        )
        
        self.conv_expand = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.conv_reduce(x)
        B, C, H, W = x.shape
        
        # Process through Swin Transformer blocks
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.conv_expand(x)
        return x

class GraphAttentionBlock(nn.Module):
    def __init__(self, channels_list, hidden_dim=256, heads=8):
        super().__init__()
        self.channels_list = channels_list
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_nodes = len(channels_list)
        
        self.positional_enc = nn.Parameter(torch.randn(1, self.num_nodes, hidden_dim))
        
        # Project each node feature to hidden_dim
        self.node_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ) for channels in channels_list
        ])
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0.2),
            GATConv(hidden_dim, hidden_dim, heads=1, dropout=0.1)
        ])
        
        # Final projections - now properly handling the channel dimensions
        self.final_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim + channels, channels, 1),  # Changed from hidden_dim*2
                nn.BatchNorm2d(channels)
            ) for channels in channels_list
        ])
        
    def create_full_adjacency(self, num_nodes):
        adj = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    adj.append([i, j])
        return torch.tensor(adj, dtype=torch.long).t().contiguous()
        
    def batch_aware_edge_index(self, edge_index, batch_size, num_nodes):
        edge_indices = []
        for i in range(batch_size):
            offset = i * num_nodes
            edge_indices.append(edge_index + offset)
        return torch.cat(edge_indices, dim=1)

    def forward(self, skip_connections):
        batch_size = skip_connections[0].size(0)
        spatial_sizes = [skip.size()[2:] for skip in skip_connections]
        
        # Project and add positional encoding
        node_features = []
        for proj, skip in zip(self.node_projections, skip_connections):
            x = proj(skip)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)
            node_features.append(x)
        
        node_features = torch.stack(node_features, dim=1) + self.positional_enc
        
        # Graph processing
        x = node_features.view(-1, self.hidden_dim)
        edge_index = self.create_full_adjacency(self.num_nodes)
        edge_index = self.batch_aware_edge_index(edge_index.to(x.device), batch_size, self.num_nodes)
        
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Residual connection
        x = x.view(batch_size, self.num_nodes, self.hidden_dim)
        output_skips = []
        for i, (proj, skip, size) in enumerate(zip(self.final_projections, skip_connections, spatial_sizes)):
            node_feat = x[:, i].view(batch_size, self.hidden_dim, 1, 1)
            node_feat = node_feat.expand(-1, -1, *size)
            output_skips.append(skip + proj(torch.cat([node_feat, skip], dim=1)))
            
        return output_skips

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        
        # Proper upsampling with channel adjustment
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, skip_channels, kernel_size=1)  # Project to skip_channels
        )
        
        # Simplified attention gate
        self.att_gate = nn.Sequential(
            nn.Conv2d(skip_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Main processing
        self.conv_block = nn.Sequential(
            nn.Conv2d(skip_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # Step 1: Upsample with channel projection
        x = self.upsample(x)
        
        # Step 2: Ensure spatial dimensions match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Step 3: Apply attention to skip connection
        att = self.att_gate(skip)
        skip = skip * att
        
        # Step 4: Concatenate and process
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)
    
class SkinLesionSegmentationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder with more channels
        self.encoder1 = EncoderBlock(in_channels, 96, dilation_rate=1)  # Increased from 64
        self.encoder2 = EncoderBlock(96, 192, dilation_rate=2)
        self.encoder3 = EncoderBlock(192, 384, dilation_rate=4)
        self.encoder4 = EncoderBlock(384, 768, dilation_rate=8)
        
        # Bottleneck with Swin Transformer
        self.swin_bottleneck = SwinTransformerBottleneck(768, 384)
        
        self.graph_att = GraphAttentionBlock([96, 192, 384, 768])
        
        # Decoder with attention
        self.decoder4 = DecoderBlock(384, 768, 768)
        self.decoder3 = DecoderBlock(768, 384, 384)
        self.decoder2 = DecoderBlock(384, 192, 192)
        self.decoder1 = DecoderBlock(192, 96, 96)
        
        # Final conv with residual
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, out_channels, 1)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 256×256×96
        e2 = self.encoder2(F.max_pool2d(e1, 2))  # 128×128×192
        e3 = self.encoder3(F.max_pool2d(e2, 2))  # 64×64×384
        e4 = self.encoder4(F.max_pool2d(e3, 2))  # 32×32×768
        
        # Bottleneck with Swin Transformer
        bottleneck = self.swin_bottleneck(F.max_pool2d(e4, 2))  # 16×16×384
        
        # Graph attention
        e1, e2, e3, e4 = self.graph_att([e1, e2, e3, e4])
        
        # Decoder
        d4 = self.decoder4(bottleneck, e4)  # in_channels=384, skip_channels=768 → out=768
        d3 = self.decoder3(d4, e3)  # in=768, skip=384 → out=384
        d2 = self.decoder2(d3, e2)  # in=384, skip=192 → out=192
        d1 = self.decoder1(d2, e1)  # in=192, skip=96 → out=96
        
        return self.final_conv(d1)

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0, alpha=0.8, gamma=2.0, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
    
    def focal_loss(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        focal_loss = self.alpha * (1 - inputs) ** self.gamma * bce
        
        return focal_loss.mean()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        # total_loss = self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal
        total_loss = self.bce_weight * bce 
        
        return total_loss
    
    @staticmethod
    def calculate_dice_score(pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    def calculate_jaccard_index(pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)

# class SkinLesionDataset(Dataset):
#     def __init__(self, data_dir, img_size=(256, 256), augment=False, preprocess=True):
#         self.image_dir = os.path.join(data_dir, 'images')
#         self.gt_dir = os.path.join(data_dir, 'gt')
#         self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
#         self.gt_files = [f.replace('.jpg', '_segmentation.png') for f in self.image_files]
#         self.img_size = img_size
#         self.augment = augment
#         self.preprocess = preprocess
        
#         # Enhanced augmentations
#         self.aug_transform = Compose([
#             RandomHorizontalFlip(p=0.5),
#             RandomVerticalFlip(p=0.5),
#             RandomRotation(degrees=20),
#             RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#             ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
#             RandomPerspective(distortion_scale=0.15, p=0.3),
#         ]) if augment else None
        
#         self.to_tensor = ToTensor()

#     def __len__(self):
#         return len(self.image_files)

#     def _preprocess_image(self, img):
#         """Modified preprocessing pipeline"""
#         img_np = np.array(img)
        
#         # 1. More conservative hair removal
#         gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Smaller kernel
#         blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
#         _, mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)  # Higher threshold
#         img_np = cv2.inpaint(img_np, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)  # Smaller radius
        
#         # 2. Milder contrast enhancement
#         lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Reduced clip limit
#         l_clahe = clahe.apply(l)
#         lab_clahe = cv2.merge((l_clahe, a, b))
#         img_np = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
#         # 3. Add gamma correction
#         gamma = 0.8
#         invGamma = 1.0 / gamma
#         table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#         img_np = cv2.LUT(img_np, table)
        
#         return Image.fromarray(img_np)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.image_files[idx])
#         gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        
#         img = Image.open(img_path).convert('RGB')
#         gt = Image.open(gt_path).convert('L')
        
#         # Apply preprocessing if enabled
#         if self.preprocess:
#             img = self._preprocess_image(img)
        
#         # Resize with appropriate interpolation
#         img = img.resize(self.img_size, Image.BICUBIC)
#         gt = gt.resize(self.img_size, Image.NEAREST)
        
#         # Synchronized augmentations
#         if self.augment:
#             seed = torch.randint(0, 2**32, (1,)).item()
#             torch.manual_seed(seed)
#             img = self.aug_transform(img)
#             torch.manual_seed(seed)
#             gt = self.aug_transform(gt)
        
#         return self.to_tensor(img), self.to_tensor(gt)

class SkinLesionDataset(Dataset):
    def __init__(self, data_dir, img_size=(256, 256), augment=False, preprocess=True):
        self.image_dir = os.path.join(data_dir, 'images')
        self.gt_dir = os.path.join(data_dir, 'gt')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.gt_files = [f.replace('.jpg', '_segmentation.png') for f in self.image_files]
        self.img_size = img_size
        self.augment = augment
        self.preprocess = preprocess
        
        # Enhanced augmentations
        self.aug_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=20),
            RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            RandomPerspective(distortion_scale=0.15, p=0.3),
        ]) if augment else None
        
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.image_files)

    def _preprocess_image(self, img):
        img_np = np.array(img)
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
        img_np = cv2.inpaint(img_np, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_np = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        gamma = 0.8
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img_np = cv2.LUT(img_np, table)
        
        return Image.fromarray(img_np)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        
        # Extract filename without extension (for saving)
        filename = os.path.splitext(self.image_files[idx])[0]
        
        if self.preprocess:
            img = self._preprocess_image(img)
        
        img = img.resize(self.img_size, Image.BICUBIC)
        gt = gt.resize(self.img_size, Image.NEAREST)
        
        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            img = self.aug_transform(img)
            torch.manual_seed(seed)
            gt = self.aug_transform(gt)
        
        return self.to_tensor(img), self.to_tensor(gt), filename

    

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_jaccard = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', leave=False)
    
    for images, gts in progress_bar:
        images, gts = images.to(device), gts.to(device)
        
        optimizer.zero_grad()
        # outputs, aux_outputs = model(images)
        
        # loss_main = criterion(outputs, gts)
        # loss_aux = criterion(aux_outputs, gts)
        # loss = loss_main + 0.5 * loss_aux
        outputs = model(images)  # Not outputs, aux_outputs
        loss = criterion(outputs, gts)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            dice_score = criterion.calculate_dice_score(outputs, gts)
            jaccard_index = criterion.calculate_jaccard_index(outputs, gts)
        
        running_loss += loss.item()
        running_dice += dice_score.item()
        running_jaccard += jaccard_index.item()
        
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'dice': running_dice / (progress_bar.n + 1),
            'jaccard': running_jaccard / (progress_bar.n + 1)
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_jaccard = running_jaccard / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_jaccard

def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_jaccard = 0.0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]', leave=False)
    
    with torch.no_grad():
        for images, gts in progress_bar:
            images, gts = images.to(device), gts.to(device)
            
            # outputs, aux_outputs = model(images)
            
            # loss_main = criterion(outputs, gts)
            # loss_aux = criterion(aux_outputs, gts)
            # loss = loss_main + 0.5 * loss_aux
            
            outputs = model(images)  # Not outputs, aux_outputs
            loss = criterion(outputs, gts)
        
            dice_score = criterion.calculate_dice_score(outputs, gts)
            jaccard_index = criterion.calculate_jaccard_index(outputs, gts)
            
            running_loss += loss.item()
            running_dice += dice_score.item()
            running_jaccard += jaccard_index.item()
            
            progress_bar.set_postfix({
                'val_loss': running_loss / (progress_bar.n + 1),
                'val_dice': running_dice / (progress_bar.n + 1),
                'val_jaccard': running_jaccard / (progress_bar.n + 1)
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_jaccard = running_jaccard / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_jaccard

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [base_lr * 0.5 * (1 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]

class EarlyStopping:
    def __init__(self, patience=100, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def main():
    batch_size = 8
    img_size = (256, 256)
    lr = 1e-4
    epochs = 200
    warmup_epochs = 5
    dataset_type = 'ISIC2017'
    
    # Loss function weights can be adjusted based on dataset
    if dataset_type == 'ISIC2018':
        bce_weight, dice_weight, focal_weight = 1.0, 0.8, 0.6
    elif dataset_type == 'REFUGE':
        bce_weight, dice_weight, focal_weight = 1.0, 1.0, 0.5
    else:
        bce_weight, dice_weight, focal_weight = 1.0, 1.0, 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dir = '../model/data 2017/train'
    val_dir = '../model/data 2017/validation'
    
    train_dataset = SkinLesionDataset(
        train_dir, 
        img_size=img_size, 
        augment=True,
        preprocess=True 
    )
    
    val_dataset = SkinLesionDataset(
        val_dir,
        img_size=img_size,
        augment=False,
        preprocess=True  
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SkinLesionSegmentationModel().to(device)
    criterion = CombinedLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        alpha=0.8,
        gamma=2.0
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_jaccard': [],
        'val_loss': [],
        'val_dice': [],
        'val_jaccard': []
    }
    
    best_val_dice = 0.0
    best_model_path = './latest/2017_best_model.pth'
    early_stopping = EarlyStopping(patience=100, verbose=True)
    
    for epoch in range(epochs):
        train_loss, train_dice, train_jaccard = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs)
        
        val_loss, val_dice, val_jaccard = validate_epoch(
            model, val_loader, criterion, device, epoch, epochs)
        
        scheduler.step()
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_jaccard'].append(train_jaccard)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_jaccard'].append(val_jaccard)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, Jaccard: {train_jaccard:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Jaccard: {val_jaccard:.4f}")
        
        if val_loss < early_stopping.val_loss_min:
            early_stopping.val_loss_min = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), './latest/2017_final_model.pth')
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_dice'], label='Train')
    plt.plot(history['val_dice'], label='Validation')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_jaccard'], label='Train')
    plt.plot(history['val_jaccard'], label='Validation')
    plt.title('Jaccard Index')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./latest/2017_training_history.png')
    plt.close()

if __name__ == "__main__":
    main()