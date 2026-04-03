import os
import csv
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model_single import TLP  # single-input model without FPG
import argparse
import csv


class PNG2DDataset(Dataset):
    def __init__(
        self,
        dir_15T,
        dir_3T,
        split="train", 
        train_split=0.8,  
        val_split=0.1,  
        seed=42,
        augment=False,
        split_csv_path=None
    ):
        super().__init__()
        self.split = split
        self.augment = augment

        self.samples = []
        self._gather_patients(dir_15T, modality="15T")
        self._gather_patients(dir_3T, modality="3T")

        def _get_pid(sample):
            # common cases: tuple/list with pid at index 0, or dict-like with 'patient_id'
            if isinstance(sample, (list, tuple)):
                return sample[0]
            return sample["patient_id"]

        # Build available patient IDs discovered on disk (with matching file lists)
        avail_ids = {s[0] for s in self.samples}

        # Patients requested by CSV for this split
        want = self.split.strip().lower()
        keep_ids = set()
        with open(split_csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            if "patient_id" not in r.fieldnames or "split" not in r.fieldnames:
                raise ValueError(f"{split_csv_path} must have columns: patient_id, split")
            for row in r:
                if row["split"].strip().lower() == want:
                    keep_ids.add(row["patient_id"].strip())

        # Show what's missing / extra
        missing_from_disk = sorted(keep_ids - avail_ids)
        extra_on_disk     = sorted(avail_ids - keep_ids)

        print(f"[split-csv] {self.split}: CSV wants {len(keep_ids)} patients; "
              f"found on disk: {len(avail_ids)}; missing: {len(missing_from_disk)}; extra: {len(extra_on_disk)}")

        # If you want to see the actual IDs (helpful while you fix paths/filenames):
        if missing_from_disk:
            print(f"[split-csv] missing IDs (on disk): {missing_from_disk[:20]}{' ...' if len(missing_from_disk)>20 else ''}")

        before = len(self.samples)
        self.samples = [s for s in self.samples if _get_pid(s) in keep_ids]
        after = len(self.samples)
        print(f"[split-csv] {self.split}: kept {after}/{before} samples using {split_csv_path}")

        # --- reduce train cohort to exactly 25×15T and 55×3T (patient-level) ---
        if self.split == "train":
            pts_15 = [s for s in self.samples if s[3] == "15T"]
            pts_3  = [s for s in self.samples if s[3] == "3T"]
            # deterministic: sort by patient_id so the same CSV → same cohort
            pts_15 = sorted(pts_15, key=lambda s: s[0])
            pts_3  = sorted(pts_3,  key=lambda s: s[0])

            '''
            need_15, need_3 = 25, 55
            if len(pts_15) < need_15 or len(pts_3) < need_3:
                print(f"[warn] requested (25;55) but available is ({len(pts_15)};{len(pts_3)}). "
                      f"Using min available.")
            chosen_15 = pts_15[:min(need_15, len(pts_15))]
            chosen_3  = pts_3[:min(need_3,  len(pts_3))]
            '''
            self.samples = pts_15 + pts_3 #chosen_15 + chosen_3
            #print(f"[train-cohort] using {len(chosen_15)}×15T and {len(chosen_3)}×3T patients (target 25;55)")


        self.slice_info = []
        for pid, reg_files, t1map_files, mod in self.samples:
            for reg_png, t1map_png in zip(sorted(reg_files), sorted(t1map_files)):
                stem = os.path.splitext(os.path.basename(reg_png))[0]
                if stem.startswith("slice_"):
                    idx = int(stem[len("slice_"):])
                    self.slice_info.append((pid, reg_png, t1map_png, idx, mod))

    def _gather_patients(self, folder_path, modality):
        if not os.path.isdir(folder_path):
            return
        for pid in os.listdir(folder_path):
            ppath = os.path.join(folder_path, pid)
            if not os.path.isdir(ppath):
                continue

            reg_folder   = os.path.join(ppath, "reg_t1w_brain")
            t1map_folder = os.path.join(ppath, "t1map_brain")
            if not (os.path.isdir(reg_folder) and os.path.isdir(t1map_folder)):
                continue

            reg_files_all = [f for f in os.listdir(reg_folder) if f.endswith(".png")]
            t1_files_all  = [f for f in os.listdir(t1map_folder) if f.endswith(".png")]

            # Use only files common to both folders (e.g., drop lone slice_241 if unmatched)
            reg_set = set(reg_files_all)
            t1_set  = set(t1_files_all)
            common  = sorted(reg_set & t1_set)  # consistent order

            if not common:
                # nothing to pair for this patient
                continue

            # Build full paths, sorted in lockstep
            reg_files = [os.path.join(reg_folder,  f) for f in common]
            t1_files  = [os.path.join(t1map_folder, f) for f in common]

            self.samples.append((pid, reg_files, t1_files, modality))


    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        pid, reg_png, t1_png, slice_idx, modality = self.slice_info[idx]

        # Always close files promptly to avoid FD leaks
        with Image.open(reg_png) as im:
            reg = np.array(im.convert("F"), dtype=np.float32)
        with Image.open(t1_png) as im:
            t1  = np.array(im.convert("F"), dtype=np.float32)

        # Normalize to [-1, 1] (match your current training)
        reg = (reg / 255.0) * 2.0 - 1.0
        t1  = (t1  / 255.0) * 2.0 - 1.0

        # (Optional) augmentations
        if getattr(self, "augment", False) and getattr(self, "split", "") == "train":
            if random.random() < 0.5:
                reg = reg[:, ::-1]
                t1  = t1[:,  ::-1]

        # Add channel dimension and convert to torch
        reg = torch.from_numpy(reg).unsqueeze(0)  # [1, H, W]
        t1  = torch.from_numpy(t1 ).unsqueeze(0)

        return reg, t1, pid, slice_idx, modality

def compute_nmse(gt, pred):
    pred = pred.astype(np.float64)
    gt   = gt.astype(np.float64)
    num = np.sum((pred - gt) ** 2)
    den = np.sum(gt ** 2) + 1e-12
    return num / den

def train(model, loader, criterion, optimizer, device, log_interval=100):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    loop = tqdm(enumerate(loader), total=len(loader), desc='Train  ', leave=False)
    for i, (x, y, pid, sidx, _modality) in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % log_interval == 0 or (i+1) == len(loader):
            loop.set_postfix(loss=loss.item())
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(loader)
    return avg_loss, epoch_time


def validate(model, loader, criterion, device, max_examples=5, out_dir=None):
    model.eval()
    total_loss = 0.0
    saved = 0
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()
    loop = tqdm(loader, total=len(loader), desc='Validate', leave=False)
    with torch.no_grad():
        # unpack the extra modality and ignore it
        for x, y, pids, sidxs, _modality in loop:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            for im, gt, pid, idx in zip(pred, y, pids, sidxs):
                if saved < max_examples:
                    arr = ((im.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                    Image.fromarray(arr).save(os.path.join(out_dir, f"pred_{pid}_s{idx}.png"))
                    saved += 1
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(loader)
    return avg_loss, epoch_time

def test(model, loader, device, out_dir):
    """
    Same behavior as before (metrics, CSV), with optional PNG saving.
    Toggle SAVE_PNGS below to enable writing predicted PNGs without changing any other functionality.
    """
    # ---------------- PNG saving options (defaults OFF) ----------------
    SAVE_PNGS = True                   # set to True to save predicted PNGs
    SAVE_INPUT_AND_GT = False           # if True, also save input and GT frames
    METHOD_NAME = "Proposed"            # subfolder name for this model
    SAVE_ROOT = os.path.join(out_dir, "PNGs")  # base folder where PNGs go
    FILE_STEM = "{:03d}"                # filename stem for slices, e.g. 000.png
    # -------------------------------------------------------------------

    def tens_to_u8_from_m11(x):
        """x: torch.Tensor [H,W] or [1,H,W] in [-1,1] -> uint8 [0,255]."""
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        x = x.detach().float().cpu().numpy()
        x = (x + 1.0) * 0.5             # [-1,1] -> [0,1]
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0 + 0.5).astype(np.uint8)
        return x

    model.eval()
    metrics = {
        "All": {"psnr":0.0, "ssim":0.0, "nmse":0.0, "count":0},
        "15T": {"psnr":0.0, "ssim":0.0, "nmse":0.0, "count":0},
        "3T":  {"psnr":0.0, "ssim":0.0, "nmse":0.0, "count":0},
    }
    slice_rows = []
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()
    loop = tqdm(loader, total=len(loader), desc='Test    ', leave=False)

    with torch.no_grad():
        for x, y, pids, sidxs, mods in loop:
            x, y = x.to(device), y.to(device)
            pred   = model(x)  # [B,1,H,W] in [-1,1] as before

            # --- metrics (unchanged) ---
            pred_np = pred.squeeze(1).cpu().numpy()
            gt_np   = y.squeeze(1).cpu().numpy()
            for i in range(pred_np.shape[0]):
                pr   = pred_np[i]
                gt   = gt_np[i]
                psnr = peak_signal_noise_ratio(gt, pr, data_range=2.0)
                ssim = structural_similarity(gt, pr, data_range=2.0)
                nmse = compute_nmse(gt, pr)
                pid  = pids[i]
                idx  = sidxs[i]
                mod  = mods[i]

                # accumulate modality & overall
                for key in ("All", mod):
                    metrics[key]["psnr"] += psnr
                    metrics[key]["ssim"] += ssim
                    metrics[key]["nmse"] += nmse
                    metrics[key]["count"] += 1

                # record slice-level
                slice_rows.append([pid, idx, mod, f"{psnr:.4f}", f"{ssim:.4f}", f"{nmse:.6f}"])

            # --- save PNGs (only if enabled) ---
            if SAVE_PNGS:
                B = pred.shape[0]
                for i in range(B):
                    pid  = str(pids[i])
                    mod  = str(mods[i])    # e.g., '1.5T' or '3T'
                    idx  = int(sidxs[i])

                    # Directory: SAVE_ROOT/METHOD_NAME/modality/patient_id/
                    out_dir_png = os.path.join(SAVE_ROOT, METHOD_NAME, mod, pid)
                    os.makedirs(out_dir_png, exist_ok=True)

                    # Save predicted slice
                    pred_u8 = tens_to_u8_from_m11(pred[i, 0])
                    Image.fromarray(pred_u8, mode="L").save(
                        os.path.join(out_dir_png, f"{FILE_STEM.format(idx)}.png")
                    )

                    if SAVE_INPUT_AND_GT:
                        inp_u8 = tens_to_u8_from_m11(x[i, 0])
                        gt_u8  = tens_to_u8_from_m11(y[i, 0])
                        Image.fromarray(inp_u8, mode="L").save(
                            os.path.join(out_dir_png, f"{FILE_STEM.format(idx)}_input.png")
                        )
                        Image.fromarray(gt_u8, mode="L").save(
                            os.path.join(out_dir_png, f"{FILE_STEM.format(idx)}_gt.png")
                        )

    total_time = time.time() - start_time

    # write per-slice CSV (unchanged)
    slice_csv = os.path.join(out_dir, "slice_metrics.csv")
    with open(slice_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id","slice_idx","modality","PSNR","SSIM","NMSE"])
        writer.writerows(slice_rows)

    # compute averages (unchanged)
    for key in metrics:
        cnt = metrics[key]["count"]
        if cnt:
            metrics[key]["psnr"] /= cnt
            metrics[key]["ssim"] /= cnt
            metrics[key]["nmse"] /= cnt

    return metrics, total_time



def main():
    parser = argparse.ArgumentParser(description="TLP train / test runner")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./outputs/7t-restormer")
    parser.add_argument("--dir15", type=str, default="./data/15T_to_7T")
    parser.add_argument("--dir3", type=str, default="./data/3T_to_7T")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--split_csv", type=str, default="./split.csv",
                    help="Path to CSV with columns: patient_id, split ∈ {train,val,test}. Always used.")
    parser.add_argument("--save-pngs", action="store_true",
                    help="If set, save predicted PNGs during test()")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    test_csv    = os.path.join(args.out_dir, "test_metrics.csv")
    val_examples= os.path.join(args.out_dir, "val_examples")
    test_results= os.path.join(args.out_dir, "test_results")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TLP(inp_A_channels=1).to(device)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state)
        print(f"▶ Loaded checkpoint: {args.ckpt}")

    train_ds = PNG2DDataset(args.dir15, args.dir3, split="train",
                             augment=True, split_csv_path=args.split_csv)
    val_ds   = PNG2DDataset(args.dir15, args.dir3, split="val",
                             augment=False, split_csv_path=args.split_csv)
    test_ds  = PNG2DDataset(args.dir15, args.dir3, split="test",
                             augment=False, split_csv_path=args.split_csv)

    common_dl_kwargs = dict(
        num_workers=args.num_workers,   # keep your CLI arg
        persistent_workers=False,       # don't keep workers alive between epochs
        prefetch_factor=1,              # minimal prefetch
        pin_memory=False                # avoid extra OS handles on some systems
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **common_dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **common_dl_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, **common_dl_kwargs)


    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.mode == "train":
        # write CSV header if new
        if not os.path.exists(metrics_csv) or os.path.getsize(metrics_csv)==0:
            with open(metrics_csv, "w", newline="") as f:
                csv.writer(f).writerow(["epoch","train_loss","val_loss"])

        best_val = float("inf")
        for ep in range(1, args.epochs+1):
            print(f"\nEpoch {ep}/{args.epochs}")
            train_loss, train_time = train(model, train_loader, criterion, optimizer, device)
            eta = train_time * (args.epochs - ep)
            print(f"  ▶ Train: loss={train_loss:.4f}, time={train_time:.1f}s, ETA={eta:.1f}s")

            val_loss, val_time = None, 0
            if ep % args.val_every == 0:
                val_loss, val_time = validate(model, val_loader, criterion, device, out_dir=val_examples)
                print(f"  ▶ Val:   loss={val_loss:.4f}, time={val_time:.1f}s")
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = os.path.join(args.out_dir, "best.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  ✔ New best, saved → {ckpt_path}")

            # always save “last”
            last_path = os.path.join(args.out_dir, "last.pth")
            torch.save(model.state_dict(), last_path)

            # log metrics
            with open(metrics_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep, f"{train_loss:.4f}",
                                 f"{val_loss:.4f}" if val_loss is not None else "N/A"])

        print("\n✅ Training complete. Running final test…")

        # fall‐through to test after training

    if args.mode in ["train", "test"]:
        metrics, test_time = test(model, test_loader, device, test_results)

        # print averages
        print(f"\n▶ Test completed in {test_time:.1f}s")
        for mod in ("All","15T","3T"):
            m = metrics[mod]
            print(f"  {mod:3s} → PSNR {m['psnr']:.4f}, SSIM {m['ssim']:.4f}, NMSE {m['nmse']:.6f}")

        # save summary CSV
        with open(test_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["modality","PSNR","SSIM","NMSE"])
            for mod in ("All","15T","3T"):
                m = metrics[mod]
                writer.writerow([mod,
                                 f"{m['psnr']:.4f}",
                                 f"{m['ssim']:.4f}",
                                 f"{m['nmse']:.6f}"])
        print(f"✅ Summary metrics saved to {test_csv}")
        print(f"✅ Slice-level metrics saved to {os.path.join(test_results,'slice_metrics.csv')}")

if __name__ == "__main__":
    main()
