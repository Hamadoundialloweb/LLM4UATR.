import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataset import UnderwaterAudioDataset
from feature_extraction import get_collate_fn
from model import MNISTMambaModelWithBEaTS


# ============ Hyperparamètres ============
HPARAMS = {
    "device_id": "3",
    "dataset_root": "/remote-home/share/dmb_nas2/Diallo/keshe/reproduction_datasets/oceanship_5s",
    "checkpoint_dir": "/remote-home/Diallo/Beat6/checkpoints5",
    "beats_checkpoint": "/remote-home/Diallo/Beat/BEATs_iter3_plus_AS2M.pt",
    
    # Features
    "feature": "mel",
    "use_beats": False,
    "beats_frozen": True,
    "fusion_strategy": "gate",
    
    # Training
    "num_classes": 6,
    "batch_size": 16,
    "num_epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "patience": 20,
    
    # Loss
    "label_smoothing": 0.1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train underwater acoustic classification model")
    parser.add_argument("--dataset_name", type=str, default="oceanship_5s")
    parser.add_argument("--feature", type=str, default="beats", choices=["mel", "mfcc", "beats"])
    parser.add_argument("--use_beats", action="store_true", help="Use BEATs features")
    parser.add_argument("--no_freeze_beats", action="store_true", help="Don't freeze BEATs encoder")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def get_device(device_id):
    """Get device for training"""
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        return torch.device("cuda")
    return torch.device("cpu")


def load_data(dataset_root, batch_size, feature, dataset_name, use_beats):
    """Load training and validation datasets"""
    train_ds = UnderwaterAudioDataset(
        root_dir=dataset_root, 
        dataset_type="train", 
        is_validation=False
    )
    val_ds = UnderwaterAudioDataset(
        root_dir=dataset_root, 
        dataset_type="test", 
        is_validation=True
    )
    
    collate = get_collate_fn(
        feature, 
        dataset_name, 
        use_beats=(feature == "beats" or use_beats)
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate, 
        num_workers=0, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate, 
        num_workers=0, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_ds.get_class_mapping()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses, correct, total = [], 0, 0
    
    for feats, labels in tqdm(loader, desc="Training", leave=False):
        feats, labels = feats.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return sum(losses) / len(losses), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    losses, correct, total = [], 0, 0
    
    for feats, labels in tqdm(loader, desc="Validation", leave=False):
        feats, labels = feats.to(device), labels.to(device)
        
        # Forward
        logits = model(feats)
        loss = criterion(logits, labels)
        
        # Metrics
        losses.append(loss.item())
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return sum(losses) / len(losses), correct / total


def main():
    args = parse_args()
    
    # Override hyperparams from command line
    if args.batch_size: 
        HPARAMS["batch_size"] = args.batch_size
    if args.epochs: 
        HPARAMS["num_epochs"] = args.epochs
    if args.lr: 
        HPARAMS["lr"] = args.lr
    
    HPARAMS["feature"] = args.feature
    HPARAMS["use_beats"] = (args.feature == "beats") or args.use_beats
    HPARAMS["beats_frozen"] = not args.no_freeze_beats
    
    # Print configuration
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Feature: {HPARAMS['feature']}")
    print(f"Use BEATs: {HPARAMS['use_beats']}")
    print(f"BEATs Frozen: {HPARAMS['beats_frozen']}")
    print(f"Batch Size: {HPARAMS['batch_size']}")
    print(f"Epochs: {HPARAMS['num_epochs']}")
    print(f"Learning Rate: {HPARAMS['lr']}")
    print("=" * 70)
    
    # Setup device
    device = get_device(HPARAMS["device_id"])
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, class_map = load_data(
        HPARAMS["dataset_root"], 
        HPARAMS["batch_size"], 
        HPARAMS["feature"], 
        args.dataset_name, 
        HPARAMS["use_beats"]
    )
    HPARAMS["num_classes"] = len(class_map)
    print(f"Number of classes: {HPARAMS['num_classes']}")
    print(f"Class mapping: {class_map}\n")
    
    # Create model
    print("Creating model...")
    model = MNISTMambaModelWithBEaTS(
        classes=HPARAMS["num_classes"],
        use_beats=HPARAMS["use_beats"],
        beats_checkpoint=HPARAMS["beats_checkpoint"],
        beats_frozen=HPARAMS["beats_frozen"],
        fusion_strategy=HPARAMS["fusion_strategy"]
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=HPARAMS["label_smoothing"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=HPARAMS["lr"], 
        weight_decay=HPARAMS["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    # Setup checkpoint saving
    save_dir = os.path.join(HPARAMS["checkpoint_dir"], args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    print(f"Checkpoints will be saved to: {save_path}\n")
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(1, HPARAMS["num_epochs"] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Log
        print(f"Epoch {epoch:3d}/{HPARAMS['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Monitor alpha
        if hasattr(model, 'alpha'):
            alpha_value = torch.sigmoid(model.alpha).item()
            print(f"           α = {alpha_value:.4f} "
                  f"(E₀: {alpha_value:.1%}, E': {(1-alpha_value):.1%})")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "hparams": HPARAMS
            }, save_path)
            
            print(f"           ✅ Best model saved! Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= HPARAMS["patience"]:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                break
        
        print("-" * 70)
    
    # Training complete
    print("=" * 70)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
