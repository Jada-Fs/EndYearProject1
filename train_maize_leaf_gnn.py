import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# STEP 0: COMPLETE `torch._utils` RESTORATION (INCLUDING `_get_obj_state`)
# =====================================================================
def _register_torch_utils():
    _mod = None
    try:
        _torch_dir = os.path.dirname(torch.__file__)
        _utils_file = os.path.join(_torch_dir, "_utils.py")
        if os.path.exists(_utils_file):
            _spec = importlib.util.spec_from_file_location("torch._utils", _utils_file)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["torch._utils"] = _mod
            _spec.loader.exec_module(_mod)
    except Exception:
        _mod = None

    if _mod is None:
        _mod = types.ModuleType("torch._utils")

    if not hasattr(_mod, "_rebuild_tensor"):
        def _rebuild_tensor(storage, storage_offset, size, stride):
            t = torch.tensor([], dtype=storage.dtype, device=storage.device)
            return t.set_(storage, storage_offset, size, stride)
        _mod._rebuild_tensor = _rebuild_tensor

    if not hasattr(_mod, "_rebuild_tensor_v2"):
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
            tensor = _mod._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        _mod._rebuild_tensor_v2 = _rebuild_tensor_v2

    if not hasattr(_mod, "_rebuild_parameter"):
        def _rebuild_parameter(data, requires_grad, backward_hooks):
            param = torch.nn.Parameter(data, requires_grad)
            param._backward_hooks = backward_hooks
            return param
        _mod._rebuild_parameter = _rebuild_parameter

    if not hasattr(_mod, "_get_obj_state"):
        def _get_obj_state(obj):
            getstate_fn = getattr(obj, "__getstate__", None)
            if getstate_fn:
                return getstate_fn()
            dict_to_save = getattr(obj, "__dict__", None)
            slots_to_save = {}
            for cls in obj.__class__.__mro__:
                if "__slots__" in cls.__dict__:
                    for slot in cls.__dict__["__slots__"]:
                        if hasattr(obj, slot):
                            slots_to_save[slot] = getattr(obj, slot)
            return (dict_to_save, slots_to_save) if slots_to_save else dict_to_save
        _mod._get_obj_state = _get_obj_state

    if not hasattr(_mod, "_get_device_index"):
        def _get_device_index(device, optional=False, allow_cpu=True):
            if isinstance(device, int):
                return device
            if isinstance(device, str):
                return 0 if "cuda" in device else -1
            if isinstance(device, torch.device):
                return 0 if device.index is None and device.type == "cuda" else (device.index if device.index is not None else -1)
            return -1
        _mod._get_device_index = _get_device_index

    if not hasattr(_mod, "ExceptionWrapper"):
        class ExceptionWrapper:
            def __init__(self, exc_info=None, where="in background"):
                self.where = where
            def reraise(self):
                raise RuntimeError(f"Error {self.where}")
        _mod.ExceptionWrapper = ExceptionWrapper

    sys.modules["torch._utils"] = _mod
    torch.__dict__["_utils"] = _mod
    setattr(torch, "_utils", _mod)


_register_torch_utils()

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =====================================================================
# PURE-PYTORCH RESNET18 BACKBONE (OUTPUT: [B, 512, 7, 7])
# =====================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class PureResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load cached ImageNet ResNet18 weights if present on disk
        try:
            cache_ckpt = os.path.expanduser(r"~\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth")
            if os.path.exists(cache_ckpt):
                state = torch.load(cache_ckpt, map_location="cpu", weights_only=False)
                filtered = {k: v for k, v in state.items() if k in self.state_dict()}
                self.load_state_dict(filtered, strict=False)
                print("Loaded cached ImageNet ResNet18 weights.")
        except Exception:
            pass

    def _make_layer(self, planes, num_blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 512, 7, 7]


# =====================================================================
# PURE-PYTORCH GATv2Conv & POOLING
# =====================================================================
class GATv2Conv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout
        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).view(num_nodes, H, C)
        x_r = self.lin_r(x).view(num_nodes, H, C)
        src, dst = edge_index[0], edge_index[1]

        alpha_raw = F.leaky_relu(x_l[dst] + x_r[src], negative_slope=0.2)
        alpha_logits = (alpha_raw * self.att).sum(dim=-1)

        max_per_dst = torch.full((num_nodes, H), -1e9, device=x.device, dtype=x.dtype)
        max_per_dst.scatter_reduce_(
            0, dst.unsqueeze(-1).expand(-1, H), alpha_logits, reduce="amax", include_self=True
        )
        alpha_exp = torch.exp(alpha_logits - max_per_dst[dst])

        denom_per_dst = torch.zeros((num_nodes, H), device=x.device, dtype=x.dtype)
        denom_per_dst.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), alpha_exp)
        alpha = alpha_exp / (denom_per_dst[dst] + 1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_r[src] * alpha.unsqueeze(-1)
        out = torch.zeros((num_nodes, H, C), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, H, C), msg)
        out = out.reshape(num_nodes, H * C) if self.concat else out.mean(dim=1)
        return out + self.bias


def global_mean_pool(x, batch):
    num_graphs = int(batch.max().item()) + 1
    out = torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
    counts = torch.zeros((num_graphs, 1), device=x.device, dtype=x.dtype)
    out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
    counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return out / counts.clamp(min=1.0)


def global_max_pool(x, batch):
    num_graphs = int(batch.max().item()) + 1
    out = torch.full((num_graphs, x.size(-1)), -1e9, device=x.device, dtype=x.dtype)
    out.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(x), x, reduce="amax", include_self=True)
    return out


# =====================================================================
# STEP 1: DATA CLEANING PIPELINE & DATA LOADERS
# =====================================================================
import hashlib

DATA_DIR = r"C:\Users\TempAdmin\Downloads\Maize_disease\Maize_disea"


def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


class CleanedImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            else:
                img = img.copy()
        return self.transform(img), label


def load_maize_data(data_dir=DATA_DIR, batch_size=32, img_size=224):
    train_tf, eval_tf = get_transforms(img_size)
    raw_dataset = datasets.ImageFolder(root=data_dir)
    class_names = raw_dataset.classes  # ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

    # --- EXPLICIT DATA CLEANING PASS ---
    # 1. Group files by MD5 hash to detect exact duplicates & cross-class label conflicts
    hash_to_items = {}
    corrupted_files = []
    non_rgb_converted = []
    resolutions = set()

    for img_path, label in raw_dataset.samples:
        try:
            with open(img_path, "rb") as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                resolutions.add(im.size)
                if im.mode != "RGB":
                    non_rgb_converted.append((os.path.basename(img_path), im.mode))
            hash_to_items.setdefault(md5_hash, []).append((img_path, label))
        except Exception:
            corrupted_files.append(img_path)

    cleaned_samples = []
    same_class_dupes_removed = 0
    cross_class_conflicts_removed = 0

    for md5_hash, items in hash_to_items.items():
        labels_set = {lbl for _, lbl in items}
        if len(labels_set) > 1:
            # Cross-class conflict (identical image in both Blight and Gray_Leaf_Spot): drop all conflicting copies
            cross_class_conflicts_removed += len(items)
        else:
            # Keep only 1 unique copy if duplicated within the same class
            cleaned_samples.append(items[0])
            if len(items) > 1:
                same_class_dupes_removed += (len(items) - 1)

    print("=" * 68)
    print("MAIZE LEAF DATASET CLEANING REPORT")
    print("=" * 68)
    print(f"Raw images scanned              : {len(raw_dataset.samples)}")
    print(f"Corrupted / unreadable removed  : {len(corrupted_files)}")
    print(f"Cross-class conflicting dupes   : {cross_class_conflicts_removed} removed (prevented label leakage)")
    print(f"Same-class exact duplicates     : {same_class_dupes_removed} removed")
    print(f"Non-RGB images standardized     : {len(non_rgb_converted)} (RGBA/CMYK -> RGB)")
    print(f"Unique raw resolutions resized  : {len(resolutions)} -> standardized to {img_size}x{img_size}")
    print(f"Clean verified images retained  : {len(cleaned_samples)}")

    # Compute per-class counts & inverse-frequency class weights for imbalance handling
    class_counts = [0] * len(class_names)
    for _, lbl in cleaned_samples:
        class_counts[lbl] += 1
    for cname, cnt in zip(class_names, class_counts):
        print(f"  - {cname:<16}: {cnt:4d} clean images ({cnt/len(cleaned_samples)*100:.1f}%)")
    print("=" * 68)

    total = len(cleaned_samples)
    val_size = int(0.15 * total)
    test_size = int(0.15 * total)
    train_size = total - val_size - test_size

    generator = torch.Generator().manual_seed(42)
    train_sub, val_sub, test_sub = random_split(
        cleaned_samples, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(CleanedImageDataset(train_sub, train_tf), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CleanedImageDataset(val_sub, eval_tf), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(CleanedImageDataset(test_sub, eval_tf), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names


# =====================================================================
# STEP 2: GRAPH CONSTRUCTION (SPATIAL GRID + DYNAMIC k-NN EDGES)
# =====================================================================
def build_grid_edge_index(h=7, w=7):
    edges = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(h):
        for c in range(w):
            u = r * w + c
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = nr * w + nc
                    edges.append((u, v))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_knn_edge_index(node_features, k=6):
    B, N, C = node_features.shape
    device = node_features.device
    norm_x = F.normalize(node_features, p=2, dim=-1)
    sim = torch.bmm(norm_x, norm_x.transpose(1, 2))
    eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(eye_mask, -1e9)
    _, topk = torch.topk(sim, k=k, dim=-1)

    offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
    src = (torch.arange(N, device=device).view(1, N, 1) + offsets).expand(B, N, k)
    dst = topk + offsets
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)


# =====================================================================
# STEP 3: HYBRID CNN + GNN MODEL
# =====================================================================
class MaizeLeafGNN(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=256, k_neighbors=6, dropout=0.3):
        super().__init__()
        self.k_neighbors = k_neighbors

        self.cnn_backbone = PureResNet18Backbone()  # [B, 512, 7, 7]
        self.register_buffer("grid_edges", build_grid_edge_index(7, 7))

        heads = 4
        self.gat1 = GATv2Conv(512, hidden_dim // heads, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images):
        feat_map = self.cnn_backbone(images)
        B, C, H, W = feat_map.shape
        N = H * W

        node_feat = feat_map.flatten(2).transpose(1, 2)

        device = images.device
        E = self.grid_edges.size(1)
        offsets = (torch.arange(B, device=device) * N).view(1, B, 1)
        spatial_edges = (self.grid_edges.unsqueeze(1).expand(2, B, E) + offsets).reshape(2, -1)
        knn_edges = build_knn_edge_index(node_feat.detach(), k=self.k_neighbors)
        edge_index = torch.cat([spatial_edges, knn_edges], dim=1)

        x = node_feat.reshape(B * N, C)
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)

        x = F.gelu(self.bn1(self.gat1(x, edge_index)))
        x = self.dropout(x)

        x_res = x
        x = F.gelu(self.bn2(self.gat2(x, edge_index)) + x_res)

        graph_embed = torch.cat([
            global_mean_pool(x, batch_vec),
            global_max_pool(x, batch_vec)
        ], dim=-1)

        return self.classifier(graph_embed)


# =====================================================================
# STEP 4: TRAIN AND EVALUATE
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = load_maize_data(DATA_DIR, batch_size=32)
    print(f"Loaded {DATA_DIR} with classes: {class_names}")

    model = MaizeLeafGNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_acc = 0.0
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_path = "best_maize_gnn.pt"
    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = (train_correct / train_total) * 100
        val_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            try:
                _register_torch_utils()
                torch.save({"model_state_dict": best_state_dict, "class_names": class_names}, save_path)
            except Exception as save_err:
                print(f"Note: Best checkpoint kept in memory ({save_err})")

    # Test evaluation using best validation weights
    model.load_state_dict(best_state_dict)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
