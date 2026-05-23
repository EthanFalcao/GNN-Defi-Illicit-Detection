"""
Plot precision-recall curve and training progress for the best GraphSAGE model
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.models.graphsage.graphsage import GraphSage, load_data, valid_indices, compute_class_weights
from src.losses.focal_loss import FocalLoss
from src.data.smote_graph import smote_augment_graph
from src.utils.set_seed import set_seed, get_trial_seed

# File paths
reports_dir = ROOT_DIR / "reports"
config_file = reports_dir / "graphsage_best_config.yaml"
base_config_file = ROOT_DIR / "configs" / "graphsage_base.yaml"

print("Loading configuration files...")
with open(config_file) as f:
    best_config = yaml.safe_load(f)

with open(base_config_file) as f:
    base_config = yaml.safe_load(f)

# Get the trial number and set seed for reproducibility
trial_num = best_config["trial"] - 1
seed = get_trial_seed(trial_num)
set_seed(seed)

print(f"\nBest Model - Trial {best_config['trial']}")
print(f"Val F1: {best_config['val']['f1']:.4f}")
print(f"Test F1: {best_config['test']['f1']:.4f}")

# Load data
print("\nLoading data...")
data, train_idx, val_idx, test_idx = load_data(
    graph_dir=ROOT_DIR / base_config["data"]["graph_dir"],
    splits_path=ROOT_DIR / base_config["data"]["splits_path"],
)

# Apply SMOTE if needed
if base_config.get("smote", {}).get("enabled", False):
    print("Applying SMOTE...")
    valid_train = train_idx[data.y[train_idx] >= 0]
    data, train_idx = smote_augment_graph(
        data, valid_train,
        k_neighbors=base_config["smote"].get("k_neighbors", 5),
        smote_k_neighbors=base_config["smote"].get("smote_k_neighbors", 5),
        random_state=seed,
    )

# Move to device
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using NVIDIA GPU (CUDA)")
else:
    device = "cpu"
    print("Using CPU")

data = data.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

# Get model config
model_config = best_config["config"]["model"]
optimizer_config = best_config["config"]["optimizer"]
training_config = best_config["config"]["training"]
loss_config = best_config["config"].get("loss", {})

# Build model
print("Building model...")
model = GraphSage(
    in_channels=data.x.size(-1),
    hidden_channels=model_config["hidden_channels"],
    out_channels=model_config["out_channels"],
    dropout=model_config["dropout"],
).to(device)

# Setup optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=optimizer_config["lr"],
    weight_decay=optimizer_config["weight_decay"],
)

# Setup loss
labelled = valid_indices(data.y, train_idx)
class_weights = compute_class_weights(data.y[labelled].cpu()).to(device)
criterion = FocalLoss(
    weight=class_weights,
    gamma=loss_config.get("gamma", 2.0),
    beta=loss_config.get("beta", 0.9999),
).to(device)

# Training loop - track metrics
epochs = training_config["epochs"]
print(f"\nTraining for {epochs} epochs...")

train_losses = []
val_losses = []
val_f1_scores = []

for epoch in range(1, epochs + 1):
    # Training
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    train_mask = valid_indices(data.y, train_idx)
    loss = criterion(logits[train_mask], data.y[train_mask])

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        val_mask = valid_indices(data.y, val_idx)

        val_loss = criterion(logits[val_mask], data.y[val_mask])
        val_losses.append(val_loss.item())

        # Calculate F1
        predictions = logits[val_mask].argmax(dim=1).cpu().numpy()
        true_labels = data.y[val_mask].cpu().numpy()
        val_f1 = f1_score(true_labels, predictions, pos_label=1, average="binary")
        val_f1_scores.append(val_f1)

    # Print progress
    if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val F1: {val_f1_scores[-1]:.4f}")

# Get final predictions for plotting
print("\nGetting predictions...")
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)

    # Validation set
    val_mask = valid_indices(data.y, val_idx)
    val_true = data.y[val_mask].cpu().numpy()
    val_probs = torch.softmax(logits[val_mask], dim=1).cpu().numpy()[:, 1]

    # Test set
    test_mask = valid_indices(data.y, test_idx)
    test_true = data.y[test_mask].cpu().numpy()
    test_probs = torch.softmax(logits[test_mask], dim=1).cpu().numpy()[:, 1]

# Plot 1: Precision-Recall Curve
print("\nPlotting precision-recall curve...")
plt.figure(figsize=(10, 8))

# Calculate precision-recall for validation
val_precision, val_recall, _ = precision_recall_curve(val_true, val_probs)
val_ap = average_precision_score(val_true, val_probs)
plt.plot(val_recall, val_precision, 'b-', linewidth=2.5, label=f'Validation (AP={val_ap:.3f})')

# Calculate precision-recall for test
test_precision, test_recall, _ = precision_recall_curve(test_true, test_probs)
test_ap = average_precision_score(test_true, test_probs)
plt.plot(test_recall, test_precision, 'r-', linewidth=2.5, label=f'Test (AP={test_ap:.3f})')

# Add baseline
val_baseline = val_true.sum() / len(val_true)
test_baseline = test_true.sum() / len(test_true)
plt.axhline(y=val_baseline, color='b', linestyle='--', alpha=0.4, label=f'Val Baseline ({val_baseline:.3f})')
plt.axhline(y=test_baseline, color='r', linestyle='--', alpha=0.4, label=f'Test Baseline ({test_baseline:.3f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Best GraphSAGE Model', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

pr_curve_file = reports_dir / "graphsage_precision_recall_curve.png"
plt.savefig(pr_curve_file, dpi=300, bbox_inches='tight')
print(f"Saved: {pr_curve_file}")
plt.close()

# Plot 2: Training Curves
print("Plotting training curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

epochs_list = list(range(1, len(train_losses) + 1))

# Loss plot
ax1.plot(epochs_list, train_losses, 'b-', linewidth=2, label='Train Loss')
ax1.plot(epochs_list, val_losses, 'r-', linewidth=2, label='Val Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# F1 plot
ax2.plot(epochs_list, val_f1_scores, 'g-', linewidth=2.5, label='Val F1')
best_f1 = max(val_f1_scores)
best_epoch = val_f1_scores.index(best_f1) + 1
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title('Validation F1 Score vs Epoch', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.0])

training_curves_file = reports_dir / "graphsage_training_curves.png"
plt.savefig(training_curves_file, dpi=300, bbox_inches='tight')
print(f"Saved: {training_curves_file}")
plt.close()

# Print summary
print("\n" + "=================================")
print(f"\nResults:")
print(f"  Val Average Precision:  {val_ap:.4f} ({val_ap*100:.2f}%)")
print(f"  Test Average Precision: {test_ap:.4f} ({test_ap*100:.2f}%)")
print(f"  Best Val F1: {best_f1:.4f} at epoch {best_epoch}")
print(f"  Final Val F1: {val_f1_scores[-1]:.4f}")

print(f"\nPlots saved to {reports_dir}/")
print("=================================")




