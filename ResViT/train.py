import os
import csv
import random
import time
import concurrent.futures
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

###################################################################################################
# 1) Dataset: Nifti2DDataset
#    - Loads volumes from 15T_to_7T (modality=0) and 3T_to_7T (modality=1).
#    - Normalizes volumes min-max, discards empty slices.
#    - Returns (input_slice, target_slice, modality_label, patient_id, slice_idx).
#########################################################################################


# ──────────────── imports (top of file) ────────────────
import os, random, collections, csv
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
# ------------------------------------------------------

class PNG2DDataset(Dataset):
    """
    Brain‑slice PNG dataset that follows a patient‑level split.csv.

    new_split.csv must have two columns:
        patient_id,split      where split ∈ {"train","val","test"} (case‑insensitive)
    """

    def __init__(
        self,
        dir_15T: str,
        dir_3T: str,
        split: str,                  # "train" | "val" | "test"
        split_csv: str,              # path to new_split.csv
        augment: bool = False,
    ):
        super().__init__()
        self.split   = split.lower()
        self.augment = augment

        # ── 0) load patient allocation from CSV ────────────────────────────
        df = pd.read_csv(split_csv, dtype={"patient_id": str})   # ← force str
        self.allowed_ids = {
            pid for pid, sp in zip(df["patient_id"], df["split"].str.lower())
            if sp == self.split
        }
        if not self.allowed_ids:
            raise RuntimeError(f"No patients with split “{split}” in {split_csv}")

        # ── 1) gather patients from both modalities ────────────────────────
        #     each entry: (patient_id, reg_png_list, t1map_png_list, modality_label)
        self.samples: list[tuple] = []
        self._gather_patients(dir_15T, modality_label=0)   # 0 = 1.5 T
        self._gather_patients(dir_3T,  modality_label=1)   # 1 = 3 T

        # keep only rows present in CSV
        self.samples = [s for s in self.samples if s[0] in self.allowed_ids]

        if len(self.samples) == 0:
            raise RuntimeError(f"No matching patients for split “{split}”. Check paths/CSV.")

        # ── 2) build slice‑level index ─────────────────────────────────────
        self.slice_info = []           # list of dicts per slice
        self.mod_counts = collections.Counter()  # slice counts per modality
        for patient_id, reg_files, t1map_files, mod_label in self.samples:
            reg_files_sorted   = sorted(reg_files)
            t1map_files_sorted = sorted(t1map_files)

            for reg_png, t1_png in zip(reg_files_sorted, t1map_files_sorted):
                stem = os.path.splitext(os.path.basename(reg_png))[0]
                if not stem.startswith("slice_"):
                    # ignore weirdly‑named files
                    continue
                slice_idx = int(stem[len("slice_"):])
                self.slice_info.append({
                    "patient_id":      patient_id,
                    "modality_label":  mod_label,      # 0 or 1
                    "reg_png":         reg_png,
                    "t1map_png":       t1_png,
                    "slice_index":     slice_idx,
                })
                self.mod_counts[mod_label] += 1

    # ───────────────────────── helpers ──────────────────────────
    def _gather_patients(self, root, modality_label):
        """Collect all patients in *root* and append to self.samples."""
        if not os.path.isdir(root):
            return
        for pid in os.listdir(root):
            ppath = os.path.join(root, pid)
            if not os.path.isdir(ppath):
                continue
            reg_folder   = os.path.join(ppath, "reg_t1w_brain")
            t1map_folder = os.path.join(ppath, "t1map_brain")
            reg_files = sorted(f for f in
                               (os.path.join(reg_folder, f) for f in os.listdir(reg_folder))
                               if f.endswith(".png"))
            t1_files  = sorted(f for f in
                               (os.path.join(t1map_folder, f) for f in os.listdir(t1map_folder))
                               if f.endswith(".png"))
            if reg_files and t1_files and [os.path.basename(x) for x in reg_files] == \
                                          [os.path.basename(x) for x in t1_files]:
                self.samples.append((pid, reg_files, t1_files, modality_label))

    # ─────────────────── torch Dataset API ─────────────────────
    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        info = self.slice_info[idx]
        pid, mod_label = info["patient_id"], info["modality_label"]
        reg = np.array(Image.open(info["reg_png"]).convert("F"))
        tgt = np.array(Image.open(info["t1map_png"]).convert("F"))

        # scale [0,255] → [-1,1]
        reg = (reg / 255.0) * 2 - 1
        tgt = (tgt / 255.0) * 2 - 1

        # optional flips
        if self.augment and self.split == "train":
            if random.random() < 0.5:
                reg, tgt = reg[:, ::-1], tgt[:, ::-1]
            if random.random() < 0.5:
                reg, tgt = reg[::-1, :], tgt[::-1, :]

        reg = np.ascontiguousarray(reg)
        tgt = np.ascontiguousarray(tgt)

        reg_t = torch.from_numpy(reg[np.newaxis]).float()
        tgt_t = torch.from_numpy(tgt[np.newaxis]).float()
        mod_t = torch.tensor([mod_label], dtype=torch.long)
        return reg_t, tgt_t, mod_t, pid, info["slice_index"]


# ───────────── sampler helper (place below the class) ─────────────
def make_balanced_sampler(ds: PNG2DDataset) -> WeightedRandomSampler:
    """
    Oversample the minority modality so that, in expectation, slices are 50/50
    across an epoch. Epoch length is kept equal to len(ds).
    """
    n_15T, n_3T = ds.mod_counts[0], ds.mod_counts[1]
    w_15T = n_3T / n_15T if n_15T else 1.0
    w_3T  = 1.0
    weights = [w_15T if info["modality_label"] == 0 else w_3T
               for info in ds.slice_info]
    return WeightedRandomSampler(weights,
                                 num_samples=len(weights),
                                 replacement=True)



###################################################################################################
# 2) Switchable 1D LayerNorm (for Transformer tokens) - no text conditioning
###################################################################################################
'''
class SwitchableLayerNorm1d(nn.Module):
    def __init__(self, hidden_size, n_modality=2, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.n_modality = n_modality
        # gamma, beta for each modality
        self.gamma = nn.Parameter(torch.ones(n_modality, hidden_size))
        self.beta  = nn.Parameter(torch.zeros(n_modality, hidden_size))

    def forward(self, x, modality_label):
        """
        x => [B, n_patches, hidden_size]
        modality_label => 0 or 1 (entire batch assumed same label).
        """
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x - mean) / (var + self.eps).sqrt()

        if isinstance(modality_label, torch.Tensor):
            modality_label = modality_label[0].item()

        gamma_ = self.gamma[modality_label]  # shape [hidden_size]
        beta_  = self.beta[modality_label]   # shape [hidden_size]

        gamma_ = gamma_.view(1, 1, -1)
        beta_  = beta_.view(1, 1, -1)

        out = x_normed * gamma_ + beta_
        return out
'''

class SwitchableLayerNorm1d(nn.Module):
    """
    LayerNorm that **ignores** modality_label – 1.5 T and 3 T share γ/β.
    Accepts the extra argument so the call‑site API stays unchanged.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x, modality_label=None):
        # x: (B, N, C) or (B, C, H, W) – last dim is the feature dim
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)

###################################################################################################
# 3) Basic Transformer Blocks (no additional conditioning)
###################################################################################################
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.transformer["num_heads"]
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out   = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_probability = config.transformer["attention_dropout_rate"]

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Torch 2.0 scaled_dot_product_attention
        context = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_probability)
        out = self.out(context)
        return out

class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DiTBlock(nn.Module):
    """
    Minimal Transformer block with switchable LN, no external conditioning.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.norm1 = SwitchableLayerNorm1d(hidden_size)
        self.norm2 = SwitchableLayerNorm1d(hidden_size)

        self.attn = Attention(config)
        self.mlp  = Mlp(config)

    def forward(self, x, modality_label):
        # LN -> self-attn
        x_norm_1 = self.norm1(x, modality_label)
        x = x + self.attn(x_norm_1)

        # LN -> MLP
        x_norm_2 = self.norm2(x, modality_label)
        x = x + self.mlp(x_norm_2)
        return x

###################################################################################################
# 4) Encoder = stack of DiTBlocks
###################################################################################################
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            DiTBlock(config) for _ in range(config.transformer["num_layers"])
        ])
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, modality_label):
        """
        x => [B, n_patches, hidden_size]
        """
        for block in self.layers:
            x = block(x, modality_label)
        x = self.final_norm(x)
        return x


###################################################################################################
# 5) Embeddings: convert 2D feature maps -> patch embeddings
###################################################################################################
class Embeddings(nn.Module):
    def __init__(self, config, n_patches, in_channels=256):
        super().__init__()
        self.patch_size   = config.patch_size            # NEW
        self.hidden_size  = config.hidden_size
        self.patch_embed  = nn.Conv2d(
            in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # we do *not* know H×W yet, so defer pos‑embed initialisation
        self.pos_embed = None
        self.dropout   = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)                      # [B, hidden, Ht, Wt]
        _, _, Ht, Wt = x.shape
        n_patches = Ht * Wt

        tokens = x.flatten(2).transpose(1, 2)        # [B, N, hidden]

        if (self.pos_embed is None) or (self.pos_embed.shape[1] != n_patches):
            self.pos_embed = nn.Parameter(
                torch.zeros(1, n_patches, self.hidden_size, device=x.device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        tokens = tokens + self.pos_embed             # ← correct rank
        return self.dropout(tokens), (Ht, Wt)



###################################################################################################
# 6) ART_block: optional "Transformer" in the middle
###################################################################################################
class ART_block(nn.Module):
    def __init__(self, config, in_channels=256, n_patches=64*64, transformer=None):
        super().__init__()
        self.transformer = transformer
        self.embeddings  = Embeddings(config, n_patches, in_channels)

        # channel compress => (in_channels + hidden_size) -> in_channels
        self.channel_compress = nn.Conv2d(in_channels + config.hidden_size, in_channels, kernel_size=1)

        # simple residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x, modality_label):
        B, _, H_map, W_map = x.shape          # ← real feature‑map size

        if self.transformer is not None:
            tokens, (Ht, Wt) = self.embeddings(x)
            out_t = self.transformer(tokens, modality_label)

            B, N, hidden = out_t.shape
            out_t = (out_t
                     .permute(0, 2, 1)
                     .reshape(B, hidden, Ht, Wt))

            # resize back to CNN resolution ── THIS LINE CHANGED
            out_t = F.interpolate(out_t,
                                  size=(H_map, W_map),
                                  mode="bilinear",
                                  align_corners=False)

            x = torch.cat([x, out_t], dim=1)
            x = self.channel_compress(x)

        # residual branch stays unchanged …
        res = x
        dx  = self.residual(x)
        return res + dx




###################################################################################################
# 7) ResViT1 model: CNN encoder -> series of ART blocks (with optional transformer) -> decoder
#    No text input, no additional conditioning. We only pass in `modality_label`.
###################################################################################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += r
        x = self.relu(x)
        return x

class ResViT1(nn.Module):
    def __init__(self, config, input_dim=1, output_dim=1):
        super().__init__()
        ngf = 64

        # Basic CNN Encoders
        enc1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, ngf, 7, padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
            #ResidualBlock(ngf),
            #ResidualBlock(ngf)
        ]
        self.encoder_1 = nn.Sequential(*enc1)

        enc2 = [
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
            #ResidualBlock(ngf*2),
            #ResidualBlock(ngf*2)
        ]
        self.encoder_2 = nn.Sequential(*enc2)

        enc3 = [
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
            #ResidualBlock(ngf*4),
            #ResidualBlock(ngf*4)
        ]
        self.encoder_3 = nn.Sequential(*enc3)

        # Define a transformer encoder
        self.transformer_encoder = Encoder(config)

        # ART blocks in bottleneck
        self.art_1 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=self.transformer_encoder)
        self.art_2 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=None)
        self.art_3 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=None)
        self.art_4 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=None)
        self.art_5 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=None)
        self.art_6 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=self.transformer_encoder)
        self.art_7 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=None)
        self.art_8 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=None)
        self.art_9 = ART_block(config, in_channels=ngf*4, n_patches=64*64, transformer=None)

        # Decoder
        dec1 = [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
            #ResidualBlock(ngf*2),
            #ResidualBlock(ngf*2)
        ]
        self.decoder_1 = nn.Sequential(*dec1)

        dec2 = [
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
            #ResidualBlock(ngf),
            #ResidualBlock(ngf)
        ]
        self.decoder_2 = nn.Sequential(*dec2)

        dec3 = [
            nn.ReflectionPad2d(3),
            #ResidualBlock(ngf),
            #ResidualBlock(ngf),
            nn.Conv2d(ngf, output_dim, 7, padding=0),
            nn.Tanh()
        ]
        self.decoder_3 = nn.Sequential(*dec3)

    def forward(self, x, modality_label):
        # Encode
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        # Bottleneck with ART blocks
        x = self.art_1(x, modality_label)
        x = self.art_2(x, modality_label)
        x = self.art_3(x, modality_label)
        x = self.art_4(x, modality_label)
        x = self.art_5(x, modality_label)
        x = self.art_6(x, modality_label)
        x = self.art_7(x, modality_label)
        x = self.art_8(x, modality_label)
        x = self.art_9(x, modality_label)

        # Decode
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x




###################################################################################################
# 8) Utility for config & main training loop (no text)
###################################################################################################
def get_config():
    import ml_collections
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.patch_size  = 4
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 4   # fewer layers for speed
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    return config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Paths
    dir_15T = "./data/15T_to_7T"
    dir_3T = "./data/3T_to_7T"
    out_dir = "./outputs/resshift"
    os.makedirs(out_dir, exist_ok=True)
    val_examples_dir = os.path.join(out_dir, "val_examples")
    os.makedirs(val_examples_dir, exist_ok=True)

    # We'll log metrics in a CSV
    metrics_file = os.path.join(out_dir, "metrics.csv")

    # Dataset / DataLoader

    train_dataset = PNG2DDataset(
        dir_15T=dir_15T,
        dir_3T=dir_3T,
        split="train",
        split_csv="./ResViT/new_split.csv",
        augment=True)

    val_dataset = PNG2DDataset(
        dir_15T=dir_15T,
        dir_3T=dir_3T,
        split="val",
        augment=False,
        split_csv="./ResViT/new_split.csv"
        )

    def list_patients(ds: PNG2DDataset, name: str):
        """Print patient IDs contained in *ds* (unique, sorted alphabetically)."""
        pids = sorted({info["patient_id"] for info in ds.slice_info})
        print(f"{name.upper():5s} split – {len(pids):3d} patient{'s' if len(pids)!=1 else ''}: "
              + (", ".join(pids) if pids else "(none)"))
        print("-" * 68)

    print("\n========== Patient allocation ==========")
    list_patients(train_dataset, "train")
    list_patients(val_dataset,   "val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model init
    config = get_config()
    model = ResViT1(config, input_dim=1, output_dim=1).to(device)
    print("Model parameters:", count_parameters(model))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    best_val_loss = float("inf")
    val_step = 5
    num_epochs = 101

    # If CSV doesn't exist or is empty, we'll need to write the header
    new_file = not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training ---
        model.train()
        epoch_loss_all = 0.0
        epoch_steps_all = 0

        for i, batch in enumerate(train_loader):
            # Each batch only has 1.5T => modality_label=0
            reg_slice_t, t1map_slice_t, modality_label_t, patient_id, slice_idx = batch
            inputs = reg_slice_t.to(device)
            labels = t1map_slice_t.to(device)
            modality_label_t = modality_label_t.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, modality_label_t)  # shape [B,1,H,W]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss_all += loss.item()
            epoch_steps_all += 1

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, step {i}, batch_loss={loss.item():.4f}")

        avg_train_loss = epoch_loss_all / max(1, epoch_steps_all)
        print(f"Epoch {epoch+1} TRAIN => 1.5T Loss: {avg_train_loss:.4f}")

        avg_val_loss = None  # We'll compute if epoch % val_step == 0

        # --- Validation ---
        if epoch % val_step == 0:
            model.eval()
            val_loss_all = 0.0
            val_steps_all = 0
            saved_examples = 0

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    reg_slice_t, t1map_slice_t, modality_label_t, patient_id, slice_idx = batch
                    inputs = reg_slice_t.to(device)
                    labels = t1map_slice_t.to(device)
                    modality_label_t = modality_label_t.to(device)

                    preds = model(inputs, modality_label_t)
                    batch_loss = criterion(preds, labels)
                    val_loss_all += batch_loss.item()
                    val_steps_all += 1

                    # Save up to 5 example slices
                    for mb_pred, mb_gt, p_id, s_idx in zip(preds, labels, patient_id, slice_idx):
                        if saved_examples < 5:
                            # convert from [-1,1] to [0,255]
                            pred_img = ((mb_pred.squeeze().cpu().numpy() + 1) / 2.0 * 255).astype(np.uint8)
                            gt_img   = ((mb_gt.squeeze().cpu().numpy() + 1) / 2.0 * 255).astype(np.uint8)

                            Image.fromarray(pred_img).save(
                                os.path.join(val_examples_dir, f"pred_epoch{epoch+1}_{p_id}_slice{s_idx}.png")
                            )
                            Image.fromarray(gt_img).save(
                                os.path.join(val_examples_dir, f"gt_epoch{epoch+1}_{p_id}_slice{s_idx}.png")
                            )
                            saved_examples += 1

            avg_val_loss = val_loss_all / max(1, val_steps_all)
            print(f"Epoch {epoch+1} VAL => 1.5T Loss: {avg_val_loss:.4f}")

            torch.save(model.state_dict(), os.path.join(out_dir, f"latest.pth"))
            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(out_dir, f"best_model.pth"))

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch+1} finished in {epoch_time:.2f} seconds")
            print("-"*50)
        else:
            # If no validation this epoch, we can still time it
            end_time = time.time()
            epoch_time = end_time - start_time

        # --- Write to CSV ---
        # We only have a 'val' loss if epoch % val_step == 0; otherwise None
        # We'll store "N/A" if we didn't do validation this epoch
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["epoch", "train_loss", "val_loss"])
                new_file = False
            val_str = f"{avg_val_loss:.4f}" if avg_val_loss is not None else "N/A"
            writer.writerow([epoch+1, f"{avg_train_loss:.4f}", val_str])


if __name__ == "__main__":
    main()
