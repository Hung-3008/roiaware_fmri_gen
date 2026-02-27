import os
import argparse
import yaml
import logging
import csv
import time
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset, Dataset
from src.utils.roi_utils import ROIDecomposer
from src.model.brain_masked_dit import BrainMaskedDiT, BrainMaskedDiTConfig
from src.model.fmri_vit_vae import FmriViTVAE, create_fmri_vit_vae
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class FmriMultiLayerDataset(Dataset):
    """Dataset pairing fMRI with multi-layer DINOv2 features."""

    def __init__(self, fmri_path, dino_path, split="train", max_samples=0):
        print(f"\nFmriMultiLayerDataset [{split}]: Loading...")
        raw_fmri = np.load(fmri_path)
        self.dino_mmap = np.load(dino_path, mmap_mode='r')

        if raw_fmri.ndim == 3:
            fmri = raw_fmri.mean(axis=1).astype(np.float32)
        elif raw_fmri.ndim == 2:
            fmri = raw_fmri.astype(np.float32)
        else:
            raise ValueError(f"Unexpected fMRI shape: {raw_fmri.shape}")
        del raw_fmri

        assert fmri.shape[0] == self.dino_mmap.shape[0], \
            f"Mismatch: fMRI {fmri.shape[0]} vs DINOv2 {self.dino_mmap.shape[0]}"

        self.fmri = fmri
        self.n_samples = fmri.shape[0]

        if max_samples > 0:
            self.fmri = self.fmri[:max_samples]
            self.n_samples = min(max_samples, self.n_samples)

        print(f"  Raw fMRI: {raw_fmri.shape if 'raw_fmri' in dir() else '(freed)'} dtype={fmri.dtype}")
        print(f"  DINOv2: {self.dino_mmap.shape} (multi-layer, mmap)")
        print(f"  {split}: {self.n_samples} samples")
        if max_samples > 0:
            print(f"  Debug: limited to {max_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).float()
        dino = torch.from_numpy(np.array(self.dino_mmap[idx])).float()
        return fmri, dino
def cosine_lr(op, ep, ev, wp, max_lr):
    if ep < wp:
        lr = max_lr * (ep + 1) / wp
    else:
        lr = max_lr * 0.5 * (1 + math.cos(math.pi * (ep - wp) / (ev - wp)))
    for pg in op.param_groups:
        pg['lr'] = lr
    return lr

def ema_update(c, e, m):
    with torch.no_grad():
        for cp, ep in zip(c.parameters(), e.parameters()):
            ep.data.mul_(m).add_(cp.data, alpha=1 - m)

def pearson_corr_voxelwise(pred, target):
    pred_zm = pred - pred.mean(0, keepdim=True)
    tgt_zm = target - target.mean(0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()

def pearson_corr_samplewise(pred, target):
    pred_zm = pred - pred.mean(1, keepdim=True)
    tgt_zm = target - target.mean(1, keepdim=True)
    num = (pred_zm * tgt_zm).sum(1)
    den = (pred_zm.norm(dim=1) * tgt_zm.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()

@torch.no_grad()
def validate(model, vae, val_loader, device, decomposer=None):
    model.eval()
    all_pred, all_true = [], []
    all_z_gen, all_z_true = [], []
    n_batches = 0
    total_val_loss = 0

    for fmri, dino in val_loader:
        fmri, dino = fmri.to(device), dino.to(device)
        z1, _, _ = vae.encode(fmri, sample_posterior=False)

        # Full masked inference: mask_ratio = 1.0 implies all fMRI tokens are masked
        z_gen, _ = model(z1, dino, mask_ratio=1.0)
        
        loss = F.mse_loss(z_gen, z1)
        total_val_loss += loss.item()
        
        fmri_pred = vae.decode(z_gen)

        all_z_gen.append(z_gen)
        all_z_true.append(z1)
        all_pred.append(fmri_pred)
        all_true.append(fmri)
        n_batches += 1

    model.train()

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)
    z_gens = torch.cat(all_z_gen)
    z_trues = torch.cat(all_z_true)

    metrics = {
        "val_loss": total_val_loss / max(n_batches, 1),
        "val_latent_mse": F.mse_loss(z_gens, z_trues).item(),
        "val_latent_pcc": pearson_corr_samplewise(z_gens, z_trues),
        "val_zgen_std": z_gens.std().item(),
        "val_zgen_crossvar_ratio": z_gens.var(dim=0).mean().item() / max(z_trues.var(dim=0).mean().item(), 1e-8),
        "val_fmri_mse": F.mse_loss(preds, trues).item(),
        "val_fmri_pcc": pearson_corr_voxelwise(preds, trues),
        "val_fmri_spcc": pearson_corr_samplewise(preds, trues),
    }

    if decomposer is not None:
        for roi in decomposer.rois:
            if roi.n_voxels > 10:
                p = preds[:, roi.indices]
                t = trues[:, roi.indices]
                metrics[f"roi_{roi.name}_spcc"] = pearson_corr_samplewise(p, t)
            else:
                metrics[f"roi_{roi.name}_spcc"] = 0.0

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Config: {cfg}")

    data_cfg = cfg["data"]
    root = data_cfg["root"]
    subject = data_cfg["subject"]
    dino_layers = cfg.get("dino_layers", [3, 6, 9, 12])
    dino_suffix = data_cfg.get("dino_suffix", "dinov2_vitb14_multilayer")

    sub_num = int(subject.replace("subj", "").lstrip("0"))
    debug_n = 128 if args.debug else 0

    train_ds = FmriMultiLayerDataset(
        os.path.join(root, subject, f"nsd_train_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject, f"nsd_{dino_suffix}_train_sub{sub_num}.npy"),
        split="train", max_samples=debug_n)
    
    test_ds = FmriMultiLayerDataset(
        os.path.join(root, subject, f"nsd_test_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject, f"nsd_{dino_suffix}_test_sub{sub_num}.npy"),
        split="test", max_samples=32 if args.debug else 0)
    train_cfg = cfg["training"]
    bs = train_cfg["batch_size"]

    if args.debug:
        train_ds = Subset(train_ds, range(128))
        test_ds = Subset(test_ds, range(32))
        bs = min(bs, 32)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Train: {len(train_ds)}, Val: {len(test_ds)}")

    # Load VAE
    vae_ckpt_path = data_cfg["vae_checkpoint"]
    vae_dir = os.path.dirname(vae_ckpt_path)
    config_path = os.path.join(vae_dir, "config.yaml")
    if not os.path.exists(config_path):
        for alt in [
            f"src/configs/{subject}/stage1_vit_vae.yaml",
            f"src/configs/exp/fmri_mlp_vae_768_{subject}.yaml",
        ]:
            if os.path.exists(alt):
                config_path = alt
                break

    with open(config_path) as f:
        vae_cfg = yaml.safe_load(f)
    if vae_cfg.get("model_type", "mlp") == "vit":
        vae = create_fmri_vit_vae(**vae_cfg["model"]).to(device).eval()
        logger.info("VAE type: ViT")
    else:
        vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
        logger.info("VAE type: MLP")

    ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"VAE loaded from {vae_ckpt_path}")

    # Load ROI Decomposer
    roi_dir = data_cfg.get("roi_dir")
    decomposer = ROIDecomposer(roi_dir) if roi_dir and os.path.exists(roi_dir) else None
    if decomposer:
        logger.info(decomposer.summary())
        roi_names = [r.name for r in decomposer.rois if r.n_voxels > 10]
    else:
        roi_names = []

    # Init Model
    model_cfg = cfg.get("model", {})
    model = BrainMaskedDiT(BrainMaskedDiTConfig(**model_cfg)).to(device)

    param_info = model.param_count()
    logger.info(f"BrainMaskedDiT: {param_info['total_M']:.1f}M params")

    use_ema = train_cfg.get("use_ema", True)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    if use_ema:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False
        logger.info("EMA enabled")
    else:
        ema_model = None

    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    num_epochs = train_cfg["num_epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    eval_interval = train_cfg.get("eval_interval", 5)
    mask_ratio = train_cfg.get("mask_ratio", 0.75)
    mask_loss_only = train_cfg.get("mask_loss_only", True)

    history_path = os.path.join(output_dir, "history.csv")
    roi_fields = [f"roi_{n}_spcc" for n in roi_names]
    fields = [
        "epoch", "train_loss", "lr", "grad_avg", "grad_max",
        "val_loss", "val_latent_mse", "val_latent_pcc",
        "val_zgen_std", "val_zgen_crossvar_ratio",
        "val_fmri_mse", "val_fmri_pcc", "val_fmri_spcc",
    ] + roi_fields
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_pcc = -1.0
    patience_counter = 0
    patience = train_cfg.get("patience", 100)

    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(optimizer, epoch - 1, num_epochs, warmup_epochs, lr)

        ep_loss, n_steps = 0, 0
        grads_all, grads_max_all = [], []
        t0 = time.time()

        for batch_idx, (fmri, dino) in enumerate(train_loader):
            fmri, dino = fmri.to(device), dino.to(device)
            B = fmri.shape[0]

            with torch.no_grad():
                z1, _, _ = vae.encode(fmri, sample_posterior=False)

            z_pred, mask = model(z1, dino, mask_ratio=mask_ratio)
            
            if mask_loss_only and mask.any():
                # Focus MSE loss only on the tokens that were masked out
                N = model.config.n_latent_tokens
                token_dim = model.token_dim
                z_pred_seq = z_pred.view(B, N, token_dim)
                z1_seq = z1.view(B, N, token_dim)
                
                loss = F.mse_loss(z_pred_seq[mask], z1_seq[mask])
            else:
                loss = F.mse_loss(z_pred, z1)

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            if use_ema:
                ema_update(model, ema_model, ema_decay)

            ep_loss += loss.item()
            grads_all.append(gn.item())
            grads_max_all.append(gn.item())
            n_steps += 1

            if batch_idx == 0 and epoch <= 5:
                logger.info(f"  [Ep{epoch} B0] loss={loss.item():.4f}")

        avg_loss = ep_loss / max(n_steps, 1)
        avg_grad = sum(grads_all) / len(grads_all) if grads_all else 0.0
        max_grad = max(grads_max_all) if grads_max_all else 0.0
        ep_time = time.time() - t0

        logger.info(f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) | loss={avg_loss:.5f} | lr={current_lr:.2e} grad={avg_grad:.4f}")

        if epoch % eval_interval == 0 or epoch == 1:
            eval_model = ema_model if use_ema else model
            val = validate(eval_model, vae, val_loader, device, decomposer=decomposer)

            row = {
                "epoch": epoch,
                "train_loss": f"{avg_loss:.6f}",
                "lr": f"{current_lr:.2e}",
                "grad_avg": f"{avg_grad:.4f}",
                "grad_max": f"{max_grad:.4f}",
                **{k: f"{v:.6f}" if 'loss' in k or 'mse' in k or 'std' in k or 'ratio' in k else f"{v:.4f}" for k, v in val.items()}
            }
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            spcc = val["val_fmri_spcc"]
            is_best = spcc > best_pcc
            logger.info(f"  VAL | loss={val['val_loss']:.4f} "
                        f"l_mse={val['val_latent_mse']:.4f} l_pcc={val['val_latent_pcc']:.4f} | "
                        f"f_mse={val['val_fmri_mse']:.4f} f_spcc={spcc:.4f}{'  ★' if is_best else ''}")
            
            roi_str = " | ".join(f"{n}={val.get(f'roi_{n}_spcc', 0):.3f}" for n in roi_names)
            logger.info(f"  ROI | {roi_str}")

            if is_best:
                best_pcc = spcc
                patience_counter = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(save_dict, os.path.join(output_dir, "best.pt"))
            else:
                patience_counter += eval_interval

            if patience > 0 and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % train_cfg.get("save_every", 50) == 0:
            torch.save({"epoch": epoch, "model_state_dict": (ema_model if use_ema else model).state_dict()}, 
                       os.path.join(output_dir, f"epoch_{epoch}.pt"))

    logger.info(f"Done! Best fMRI sPCC: {best_pcc:.4f}")

if __name__ == "__main__":
    main()
