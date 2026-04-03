import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from PIL import Image

# Import the updated dataloader, model and config
from train import PNG2DDataset, ResViT1, get_config  # adapt as needed

def print_split_patients(tag: str, ds: PNG2DDataset) -> None:
    """
    Args
    ----
    tag : str
        A label such as 'TRAIN', 'VAL', or 'TEST' for the header.
    ds  : PNG2DDataset
        An already‑constructed dataset instance.

    Prints
    ------
    • Count of patients in the subset  
    • Comma‑separated list of their IDs (alphabetical order)
    """
    pids = sorted({pid for pid, *_ in ds.samples})  # ds.samples is patient‑level
    print(f"\n{tag} split  ─ {len(pids)} patients\n" + ", ".join(pids) + "\n")


def compute_nmse(pred, gt):
    pred = pred.astype(np.float64)
    gt   = gt.astype(np.float64)
    num = np.sum((pred - gt) ** 2)
    den = np.sum(gt ** 2) + 1e-12
    return num / den

def compute_ncc(pred, gt):
    pm, gm = pred.mean(), gt.mean()
    num = np.sum((pred - pm)*(gt - gm))
    den = np.sqrt(np.sum((pred - pm)**2) * np.sum((gt - gm)**2)) + 1e-12
    return num/den

def test_slice_by_slice(
    dir_15T, 
    dir_3T,
    checkpoint_path,
    split_csv="new_split.csv",
    out_csv="test_metrics_slice_by_slice.csv",
    batch_size=4,
    device="cuda",
):
    # --- prepare data & model ---
    test_dataset = PNG2DDataset(
        dir_15T=dir_15T,
        dir_3T=dir_3T,
        split="test",
        split_csv=split_csv,
        augment=False,
    )

    print_split_patients("TEST", test_dataset)


    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    config = get_config()
    model  = ResViT1(config, input_dim=1, output_dim=1).to(device)

    # ------------------------------------------------------------------
    # Warm‑start once so art_*.embeddings.pos_embed exists, *then* load
    # ------------------------------------------------------------------
    reg0, _, mod0, *_ = test_dataset[0]          # sample slice → get H, W and a label
    H, W = reg0.shape[-2:]
    with torch.no_grad():                         # create the two pos_embed tensors
        _ = model(torch.zeros(1, 1, H, W, device=device),
                  torch.tensor([mod0.item()], device=device))

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)             # strict=True now succeeds
    model.eval()


    # --- buckets for metrics ---
    metrics = {
        "All":  {"psnr": [], "ssim": [], "nmse": [], "ncc": []},
        0:      {"psnr": [], "ssim": [], "nmse": [], "ncc": []},  # 1.5T
        1:      {"psnr": [], "ssim": [], "nmse": [], "ncc": []},  # 3T
    }

    # header + rows for per-slice CSV
    rows = []
    header = ["patient_id","slice_idx","modality","PSNR","SSIM","NMSE","NCC"]

    # --- inference & metrics collection ---
    with torch.no_grad():
        for reg_t, t1_t, mod_t, pids, sidxs in test_loader:
            reg_t = reg_t.to(device)           # [B,1,H,W]
            t1_t  = t1_t.to(device)
            mod_t = mod_t.to(device)           # [B] or [B,1]

            preds = model(reg_t, mod_t)        # [B,1,H,W]
            pred_np = preds[:,0].cpu().numpy()
            gt_np   = t1_t[:,0].cpu().numpy()

            for i in range(pred_np.shape[0]):
                gt = gt_np[i]
                if not np.any(gt):
                    continue  # skip empty GT slices

                pr   = pred_np[i]
                ps   = peak_signal_noise_ratio(gt, pr, data_range=2.0)
                ss   = ssim(gt, pr, data_range=2.0)
                nm   = compute_nmse(pr, gt)
                nc   = compute_ncc(pr, gt)
                pid  = pids[i]
                sidx = int(sidxs[i].item())
                m    = int(mod_t[i].item())

                # accumulate
                for key in ("All", m):
                    metrics[key]["psnr"].append(ps)
                    metrics[key]["ssim"].append(ss)
                    metrics[key]["nmse"].append(nm)
                    metrics[key]["ncc"].append(nc)

                # record slice row
                mod_str = "1.5T" if m==0 else "3T"
                rows.append([pid, sidx, mod_str,
                             f"{ps:.4f}", f"{ss:.4f}",
                             f"{nm:.6f}", f"{nc:.4f}"])

                # save PNG
                '''
                img8 = ((pr + 1)/2*255).clip(0,255).astype(np.uint8)
                out_dir = os.path.join(f"./{mod_str}_new_pngs", pid)
                os.makedirs(out_dir, exist_ok=True)
                Image.fromarray(img8).save(
                    os.path.join(out_dir, f"slice_{sidx:03d}.png")
                )
                '''
                
                
    # --- write per-slice CSV once ---
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # --- compute & print + save summary stats ---
    summary_csv = os.path.splitext(out_csv)[0] + "_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "modality",
            "mean_psnr","std_psnr",
            "mean_ssim","std_ssim",
            "mean_nmse","std_nmse",
            "mean_ncc","std_ncc"
        ])
        for key in ("All",0,1):
            arrs = metrics[key]
            if not arrs["psnr"]:
                continue
            mean_psnr, std_psnr = np.mean(arrs["psnr"]), np.std(arrs["psnr"])
            mean_ssim, std_ssim = np.mean(arrs["ssim"]), np.std(arrs["ssim"])
            mean_nmse, std_nmse = np.mean(arrs["nmse"]), np.std(arrs["nmse"])
            mean_ncc,  std_ncc  = np.mean(arrs["ncc"]),  np.std(arrs["ncc"])
            mod_str = key if key=="All" else ("1.5T" if key==0 else "3T")

            # print to terminal
            print(f"\n--- {mod_str} ---")
            print(f"PSNR: {mean_psnr:.4f} ± {std_psnr:.4f}")
            print(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
            print(f"NMSE: {mean_nmse:.6f} ± {std_nmse:.6f}")
            print(f"NCC:  {mean_ncc:.4f} ± {std_ncc:.4f}")

            # save to summary CSV
            writer.writerow([
                mod_str,
                f"{mean_psnr:.4f}", f"{std_psnr:.4f}",
                f"{mean_ssim:.4f}", f"{std_ssim:.4f}",
                f"{mean_nmse:.6f}", f"{std_nmse:.6f}",
                f"{mean_ncc:.4f}",  f"{std_ncc:.4f}",
            ])

    print(f"\n✅ Slice metrics → {out_csv}")
    print(f"✅ Summary metrics → {summary_csv}")

# example invocation
if __name__ == "__main__":
    test_slice_by_slice(
        dir_15T="./data/15T_to_7T",
        dir_3T="./data/3T_to_7T",
        checkpoint_path="./outputs/resshift/best_model.pth",
        split_csv="./ResViT/new_split.csv",
        out_csv="./outputs/resshift/test_metrics_slice_by_slice.csv",
        batch_size=8,
        device="cuda"
    )
