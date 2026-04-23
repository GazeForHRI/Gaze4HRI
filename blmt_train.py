import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import wandb
import argparse
import config
from h5_blink4hri import Blink4HRITorchDataset
from blinklinmult.models import BlinkLinMulT

def create_dataloaders(batch_size=64, n_frames=15, num_workers=4):
    base_dir = config.get_dataset_base_directory()
    split_path = os.path.join(base_dir, "blink4hri_torch_dataset", "split.json")
    stats_path = os.path.join(base_dir, "blink4hri_torch_dataset", "feature_stats.json")
    
    with open(split_path, "r") as f:
        splits = json.load(f)

    # Load standardization stats
    mean_std_dict = None
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            mean_std_dict = json.load(f)
    else:
        print("WARNING: feature_stats.json not found! High-level features will NOT be standardized.")

    train_ds = Blink4HRITorchDataset(splits["train"], n_frames=n_frames, mean_std_dict=mean_std_dict)
    val_ds = Blink4HRITorchDataset(splits["val"], n_frames=n_frames, mean_std_dict=mean_std_dict)
    
    # Calculate positive weight for BCEWithLogitsLoss to handle imbalance
    train_labels = np.array(train_ds.get_labels())
    num_positive = np.sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32) if num_positive > 0 else None
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, pos_weight

def train_model(epochs=50, batch_size=64, lr=1e-3, weight_decay=5e-4, weights='none', run_name="blmt_training"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine weight loading strategy
    is_checkpoint_path = os.path.isfile(weights)
    if is_checkpoint_path:
        model_weights = None
        print(f"Loading custom checkpoint from: {weights}")
    else:
        model_weights = None if weights.lower() == 'none' else weights

    # Initialize Weights & Biases
    wandb.init(
        project="blink4hri",
        name=run_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "architecture": "BlinkLinMulT",
            "dataset": "blink4hri",
            "init_weights": weights if is_checkpoint_path else str(model_weights)
        }
    )

    train_loader, val_loader, pos_weight = create_dataloaders(batch_size=batch_size)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    # Initialize model
    model = BlinkLinMulT(input_dim=160, output_dim=1, weights=model_weights)
    
    # If a path was provided, load the state dict manually
    if is_checkpoint_path:
        model.load_state_dict(torch.load(weights, map_location=device))
        
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    best_val_f1 = 0.0
    # Set save directory using wandb run id
    save_dir = os.path.join(config.get_dataset_base_directory(), "blink4hri_torch_dataset", "checkpoints", wandb.run.id)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            left_eyes, left_feats = batch["left_eye"].to(device), batch["left_features"].to(device)
            right_eyes, right_feats = batch["right_eye"].to(device), batch["right_features"].to(device)
            blinks_frame = batch["is_blink"].to(device)
            
            optimizer.zero_grad()
            
            out_l = model([left_eyes, left_feats])
            out_r = model([right_eyes, right_feats])
            
            y_preds_l = out_l[1] if isinstance(out_l, tuple) else out_l
            y_preds_r = out_r[1] if isinstance(out_r, tuple) else out_r
            
            avg_logits = (y_preds_l.squeeze(-1) + y_preds_r.squeeze(-1)) / 2.0
            loss = criterion(avg_logits, blinks_frame)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            probs = torch.sigmoid(avg_logits)
            preds = (probs > 0.5).float()
            
            train_preds.extend(preds.view(-1).cpu().numpy())
            train_targets.extend(blinks_frame.view(-1).cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_prec = precision_score(train_targets, train_preds, zero_division=0)
        train_rec = recall_score(train_targets, train_preds, zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        train_iou = jaccard_score(train_targets, train_preds, zero_division=0)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                left_eyes, left_feats = batch["left_eye"].to(device), batch["left_features"].to(device)
                right_eyes, right_feats = batch["right_eye"].to(device), batch["right_features"].to(device)
                blinks_frame = batch["is_blink"].to(device)
                
                out_l = model([left_eyes, left_feats])
                out_r = model([right_eyes, right_feats])
                
                y_preds_l = out_l[1] if isinstance(out_l, tuple) else out_l
                y_preds_r = out_r[1] if isinstance(out_r, tuple) else out_r
                
                avg_logits = (y_preds_l.squeeze(-1) + y_preds_r.squeeze(-1)) / 2.0
                loss = criterion(avg_logits, blinks_frame)
                val_loss += loss.item()
                
                probs = torch.sigmoid(avg_logits)
                preds = (probs > 0.5).float()
                
                val_preds.extend(preds.view(-1).cpu().numpy())
                val_targets.extend(blinks_frame.view(-1).cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec = precision_score(val_targets, val_preds, zero_division=0)
        val_rec = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_iou = jaccard_score(val_targets, val_preds, zero_division=0)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        # --- LOGGING ---
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/acc": train_acc,
            "train/precision": train_prec,
            "train/recall": train_rec,
            "train/f1": train_f1,
            "train/iou": train_iou,
            "val/loss": avg_val_loss,
            "val/acc": val_acc,
            "val/precision": val_prec,
            "val/recall": val_rec,
            "val/f1": val_f1,
            "val/iou": val_iou,
            "lr": current_lr
        })
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} F1: {train_f1:.4f} IoU: {train_iou:.4f} | Val Loss: {avg_val_loss:.4f} F1: {val_f1:.4f} IoU: {val_iou:.4f} | LR: {current_lr}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(save_dir, "best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"-> Saved new best model to {save_path} (Val F1: {best_val_f1:.4f})")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BlinkLinMulT")
    parser.add_argument("--weights", type=str, default="none", 
                        help="Weights to load. Use 'none', 'blinklinmult-union', or an absolute path.")
    parser.add_argument("--run_name", type=str, default=None, 
                        help="Name of the run for Weights & Biases. Required if loading from path.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    
    args = parser.parse_args()

    # Handle automatic run_name logic
    if args.run_name is None:
        if args.weights.lower() == "none":
            args.run_name = "blmt_random_init"
        elif args.weights.lower() == "blinklinmult-union":
            args.run_name = "blmt_union_init"
        else:
            parser.error("--run_name is required when providing an absolute path to --weights")
    
    train_model(epochs=args.epochs, weights=args.weights, run_name=args.run_name)
