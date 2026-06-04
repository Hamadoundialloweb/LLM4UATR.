import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import pandas as pd
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataset import UnderwaterAudioDataset
from feature_extraction import get_collate_fn
from model import MNISTMambaModelWithBEaTS


# ============ Hyperparamètres ============
HPARAMS = {
    "device_id": "1",
    "dataset_root": "/remote-home/share/dmb_nas2/Diallo/keshe/reproduction_datasets/oceanship_5s",
    "checkpoint_path": "/remote-home/Diallo/Beat6/checkpoints3/oceanship_5s/best_model.pth",
    "beats_checkpoint": "/remote-home/Diallo/Beat/BEATs_iter3_plus_AS2M.pt",
    "result_dir": "/remote-home/Diallo/Beat6/test3_results",
    
    # Features
    "feature": "mel",
    "use_beats": true,
    "beats_frozen": True,
    "fusion_strategy": "gate",
    
    # Test
    "batch_size": 16,
    "num_classes": 6,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test underwater acoustic classification model")
    parser.add_argument("--dataset_name", type=str, default="oceanship_5s")
    parser.add_argument("--feature", type=str, default="beats", choices=["mel", "mfcc", "beats"])
    parser.add_argument("--use_beats", action="store_true", help="Use BEATs features")
    parser.add_argument("--no_freeze_beats", action="store_true", help="Don't freeze BEATs encoder")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    return parser.parse_args()


def get_device(device_id):
    """Setup device for testing"""
    if device_id and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device("cuda")
        print(f"✅ Using GPU: {device_id}")
        return device
    print("✅ Using CPU")
    return torch.device("cpu")


def load_test_data(dataset_root, batch_size, feature, dataset_name, use_beats):
    """Load test dataset"""
    print(f"\n📦 Loading test dataset...")
    
    test_dataset = UnderwaterAudioDataset(
        root_dir=dataset_root,
        dataset_type="test",
        is_validation=True
    )
    
    collate_fn = get_collate_fn(
        feature, 
        dataset_name, 
        use_beats=(feature == "beats" or use_beats)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    class_to_idx = test_dataset.get_class_mapping()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print(f"   Samples: {len(test_dataset)}")
    print(f"   Classes: {len(class_to_idx)}")
    print(f"   Class names: {list(class_to_idx.keys())}\n")
    
    return test_loader, class_to_idx, idx_to_class


def init_model(num_classes, device, hparams):
    """Initialize model and load checkpoint"""
    print("🏗️  Initializing model...")
    
    model = MNISTMambaModelWithBEaTS(
        classes=num_classes,
        use_beats=hparams["use_beats"],
        beats_checkpoint=hparams["beats_checkpoint"],
        beats_frozen=hparams["beats_frozen"],
        fusion_strategy=hparams["fusion_strategy"]
    ).to(device)

    ckpt_path = hparams["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")

    print(f"\n📂 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get('epoch', 'N/A')
        best_acc = checkpoint.get('best_val_acc', 'N/A')
        print(f"   Checkpoint info:")
        print(f"   - Epoch: {epoch}")
        print(f"   - Best Val Acc: {best_acc:.4f}" if isinstance(best_acc, float) else f"   - Best Val Acc: {best_acc}")
    else:
        state_dict = checkpoint
        print(f"   Loading state dict directly (legacy format)")
    
    # Handle DataParallel models
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys: {missing}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected}")
    
    model.eval()
    print("✅ Model loaded successfully\n")
    
    return model


@torch.no_grad()
def infer(model, test_loader, device):
    """Run inference on test set"""
    print("🔍 Running inference...")
    model.eval()
    
    y_true = []
    y_pred = []
    
    # Track alpha if available
    alpha_values = []
    
    pbar = tqdm(test_loader, desc="Testing")
    for feats, labels in pbar:
        feats, labels = feats.to(device), labels.to(device)
        
        # Forward pass
        logits = model(feats)
        preds = logits.argmax(dim=1)
        
        # Collect predictions
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        
        # Track alpha
        if hasattr(model, 'alpha'):
            alpha_values.append(torch.sigmoid(model.alpha).item())
        
        # Update progress bar
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            pbar.set_postfix({'acc': f'{acc:.4f}'})
    
    print("✅ Inference complete\n")
    
    results = {
        'y_true': np.array(y_true),
        'y_pred': np.array(y_pred)
    }
    
    # Add alpha info if available
    if alpha_values:
        avg_alpha = np.mean(alpha_values)
        results['alpha'] = avg_alpha
        print(f"📊 Alpha value: {avg_alpha:.4f}")
        print(f"   E₀ contribution: {avg_alpha:.1%}")
        print(f"   E' contribution: {(1-avg_alpha):.1%}\n")
    
    return results


def compute_metrics(results, idx_to_class, result_dir):
    """Compute and save metrics"""
    print("=" * 70)
    print("📈 COMPUTING METRICS")
    print("=" * 70)
    
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print(f"\n📊 OVERALL METRICS")
    print(f"   Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
    print(f"   Precision:    {prec:.4f}")
    print(f"   Recall:       {recall:.4f}")
    print(f"   F1-macro:     {f1_macro:.4f}")
    print(f"   F1-weighted:  {f1_weighted:.4f}")
    
    # Classification report
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names, 
        output_dict=True, 
        zero_division=0
    )
    
    # Print per-class metrics
    print(f"\n📊 PER-CLASS METRICS")
    for class_name in target_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"   {class_name:15s} - P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}  F1: {metrics['f1-score']:.4f}")
    
    # Save results
    os.makedirs(result_dir, exist_ok=True)
    
    # Save classification report
    df_report = pd.DataFrame(report).transpose()
    csv_path = os.path.join(result_dir, "classification_report.csv")
    df_report.to_csv(csv_path)
    print(f"\n✅ Classification report saved: {csv_path}")
    
    # Save metrics JSON
    import json
    metrics_dict = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(recall),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            class_name: {
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1-score": float(report[class_name]["f1-score"]),
                "support": int(report[class_name]["support"])
            }
            for class_name in target_names if class_name in report
        }
    }
    
    # Add alpha if available
    if 'alpha' in results:
        metrics_dict['alpha'] = {
            'value': float(results['alpha']),
            'E0_contribution': float(results['alpha']),
            'Eprime_contribution': float(1 - results['alpha'])
        }
    
    json_path = os.path.join(result_dir, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✅ Metrics JSON saved: {json_path}")
    
    return acc, f1_macro, cm


def plot_confusion_matrix(cm, idx_to_class, save_path):
    """Plot and save confusion matrix"""
    labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    # Normalize confusion matrix
    cmn = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Raw counts
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        xticklabels=labels, 
        yticklabels=labels, 
        cmap="Blues", 
        ax=axes[0], 
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=14, fontweight='bold')
    
    # Normalized percentages
    sns.heatmap(
        cmn, 
        annot=True, 
        fmt=".1f", 
        xticklabels=labels, 
        yticklabels=labels, 
        cmap="Reds", 
        ax=axes[1], 
        cbar_kws={'label': 'Percentage (%)'}
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')
    
    # Format axes
    for ax in axes:
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


def save_summary(results, hparams, class_to_idx, save_path):
    """Save test summary to text file"""
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TEST RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Dataset: {hparams.get('dataset_root', 'N/A')}\n")
        f.write(f"  Checkpoint: {hparams.get('checkpoint_path', 'N/A')}\n")
        f.write(f"  Feature: {hparams.get('feature', 'N/A')}\n")
        f.write(f"  Use BEATs: {hparams.get('use_beats', 'N/A')}\n")
        f.write(f"  Batch Size: {hparams.get('batch_size', 'N/A')}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  Total Samples: {len(y_true)}\n")
        f.write(f"  Number of Classes: {len(class_to_idx)}\n")
        f.write(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  F1-macro: {f1:.4f}\n")
        
        if 'alpha' in results:
            alpha = results['alpha']
            f.write(f"\nALPHA RESIDUAL:\n")
            f.write(f"  α value: {alpha:.4f}\n")
            f.write(f"  E₀ contribution: {alpha:.1%}\n")
            f.write(f"  E' contribution: {(1-alpha):.1%}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Summary saved: {save_path}")


def main():
    args = parse_args()
    
    # Override hyperparams from command line
    if args.batch_size: 
        HPARAMS["batch_size"] = args.batch_size
    if args.checkpoint_path: 
        HPARAMS["checkpoint_path"] = args.checkpoint_path
    if args.result_dir: 
        HPARAMS["result_dir"] = args.result_dir

    HPARAMS["feature"] = args.feature
    HPARAMS["use_beats"] = (args.feature == "beats") or args.use_beats
    HPARAMS["beats_frozen"] = not args.no_freeze_beats
    
    # Print configuration
    print("\n" + "=" * 70)
    print("🧪 TEST CONFIGURATION")
    print("=" * 70)
    print(f"📁 Dataset: {args.dataset_name}")
    print(f"🎵 Feature: {HPARAMS['feature']}")
    print(f"🔬 Use BEATs: {HPARAMS['use_beats']}")
    print(f"🔒 BEATs Frozen: {HPARAMS['beats_frozen']}")
    print(f"📦 Batch Size: {HPARAMS['batch_size']}")
    print(f"📂 Checkpoint: {HPARAMS['checkpoint_path']}")
    print(f"📊 Results Dir: {HPARAMS['result_dir']}")
    print("=" * 70)
    
    # Setup device
    device = get_device(HPARAMS["device_id"])

    # Load test data
    test_loader, class_to_idx, idx_to_class = load_test_data(
        HPARAMS["dataset_root"],
        HPARAMS["batch_size"],
        HPARAMS["feature"],
        args.dataset_name,
        HPARAMS["use_beats"]
    )

    # Initialize model
    HPARAMS["num_classes"] = len(class_to_idx)
    model = init_model(HPARAMS["num_classes"], device, HPARAMS)
    
    # Run inference
    results = infer(model, test_loader, device)

    # Compute metrics
    acc, f1, cm = compute_metrics(results, idx_to_class, HPARAMS["result_dir"])
    
    # Plot confusion matrix
    cm_path = os.path.join(HPARAMS["result_dir"], "confusion_matrix.png")
    plot_confusion_matrix(cm, idx_to_class, cm_path)
    
    # Save summary
    summary_path = os.path.join(HPARAMS["result_dir"], "test_summary.txt")
    save_summary(results, HPARAMS, class_to_idx, summary_path)
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS")
    print("=" * 70)
    print(f"   Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1-macro:     {f1:.4f}")
    print(f"   Total Samples: {len(results['y_true'])}")
    print(f"   Classes:      {len(class_to_idx)}")
    if 'alpha' in results:
        print(f"   Alpha (α):    {results['alpha']:.4f}")
    print("=" * 70)
    print(f"✅ All results saved to: {HPARAMS['result_dir']}\n")


if __name__ == "__main__":
    main()
