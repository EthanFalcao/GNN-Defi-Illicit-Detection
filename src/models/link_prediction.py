import numpy as np
import torch
import torch.nn.functional as F
import argparse

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import random

CODE_SEED = 42

torch.manual_seed(CODE_SEED)
np.random.seed(CODE_SEED)
random.seed(CODE_SEED)
torch.cuda.manual_seed_all(CODE_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------
# LOAD DATA
# ---------------

X = np.load("data/processed/graph/node_features.npy")
edge_index = np.load("data/processed/graph/edge_index.npy")

data = Data(x = torch.tensor(X, dtype=torch.float),
            edge_index = torch.tensor(edge_index, dtype=torch.long))


# ----------------------
# HELPER FUNCTIONS
# ----------------------
def compute_f1(labels, probs):
   thresholds = np.linspace(0, 1, 100)
   best_f1 = 0
   best_thresh = 0.5
            
   for t in thresholds:
     preds = (probs >= t).astype(int)
     f1 = f1_score(labels, preds)
     if f1 > best_f1:
       best_f1 = f1
       best_thresh = t
   return best_f1, best_thresh

def plot_score_distribution(probs, labels):
    import matplotlib.pyplot as plt

    plt.figure()

    plt.hist(probs[labels == 1], bins=50, alpha=0.6, label="Positive edges")
    plt.hist(probs[labels == 0], bins=50, alpha=0.6, label="Negative edges")

    plt.title("Edge Score Distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

def plot_curves(labels, probs):
    import matplotlib.pyplot as plt

    # ROC
    fpr, tpr, _ = roc_curve(labels, probs)

    # PR
    precision, recall, _ = precision_recall_curve(labels, probs)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.subplot(1,2,2)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.tight_layout()
    plt.show()

# ------------------
# GRAPHSAGE MODEL
# ------------------

class GraphSAGE(torch.nn.Module):
  def __init__(self, in_channels, hidden_dim, dropout):
    super().__init__()
    self.conv1 = SAGEConv(in_channels, hidden_dim)
    self.conv2 = SAGEConv(hidden_dim, hidden_dim)
    self.dropout = torch.nn.Dropout(p=dropout)

  def encode(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = self.dropout(x)            # added dropout
    x = self.conv2(x, edge_index)
    return x
    
  def decode(self, z, edge_label_index):
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=1)

  def forward(self, x, edge_index, edge_label_index):
    z = self.encode(x, edge_index)
    out = self.decode(z, edge_label_index)
    return out

# ---------------
# SPLIT EDGES
# ---------------
split = RandomLinkSplit(
      num_val = 0.1,
      num_test = 0.1,
      is_undirected = False,
      add_negative_train_samples=True,
      neg_sampling_ratio=1
  )
train_data, val_data, test_data = split(data)

# Main link prediction code
def run_link_prediction(args, train_data, val_data, test_data, return_val_only=False):
  best_val_ap = 0
  train_losses = []
  val_aucs = []
  val_aps = []
  
  

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_data = train_data.to(device)
  val_data = val_data.to(device)
  test_data = test_data.to(device)

  in_channels = X.shape[1]
  model = GraphSAGE(in_channels, args.hidden_dim, args.dropout).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)            # added weight decay
  loss_fn = torch.nn.BCEWithLogitsLoss()  #using BCE loss

  @torch.no_grad()
  def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
  
    probs = torch.sigmoid(out).cpu().numpy()
    labels = data.edge_label.cpu().numpy()
  
    roc_auc = roc_auc_score(labels, probs)
    avg_pred_score = average_precision_score(labels, probs)

    f1, thresh = compute_f1(labels, probs)
  
    return roc_auc, avg_pred_score, f1, thresh, probs, labels
              
  best_model_state = None
            
  # training loop  
  for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(
            train_data.x,
            train_data.edge_index,
            train_data.edge_label_index
        )

        loss = loss_fn(out, train_data.edge_label.float())
        loss.backward()
        optimizer.step()

        val_auc, val_ap, val_f1, val_thresh, val_probs, val_labels = evaluate(model, val_data)

        if val_ap > best_val_ap:
           best_val_ap = val_ap
           best_model_state = model.state_dict()
        
        train_losses.append(loss.item())
        val_aucs.append(val_auc)
        val_aps.append(val_ap)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Val AUC {val_auc:.4f} | "
            f"Val AP {val_ap:.4f}",
            f"Val F1 {val_f1:.4f}",        
        )
  
  model.load_state_dict(best_model_state)
  
  if return_val_only:
    return best_val_ap
              
  print("Training curve")
  plt.figure()
  plt.plot(train_losses)
  plt.title("Training Loss")
  plt.show()
  
  plt.figure()
  plt.plot(val_aucs, label="AUC")
  plt.plot(val_aps, label="AP")
  plt.legend()
  plt.title("Validation Metrics")
  plt.show()

  test_auc, test_ap, test_f1, test_thresh, test_probs, test_labels = evaluate(model, test_data)

  print("\nFINAL TEST RESULTS")
  print("ROC-AUC:", test_auc)
  print("Average Precision Score:", test_ap)
  print("F1:", test_f1)
  print("Best Threshold:", test_thresh)

  # VISUALIZATIONS
 
  # test curves
  plot_curves(test_labels, test_probs)
  plot_score_distribution(test_probs, test_labels)
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()

    best_config = None
    best_val_ap = 0

    # hyper parameter tuning        
    for hidden_dim in [64, 128]:
      for lr in [0.001, 0.005]:
            for dropout in [0.2, 0.5]:
                    torch.manual_seed(CODE_SEED)
                    np.random.seed(CODE_SEED)
                    random.seed(CODE_SEED)
                        
                    print(f"\nRunning hidden_dim={hidden_dim}, lr={lr},  dropout={dropout}")
            
                    args.hidden_dim = hidden_dim
                    args.lr = lr
                    args.dropout = dropout
            
                    val_ap = run_link_prediction(args, train_data, val_data, test_data, return_val_only=True)

                    if val_ap > best_val_ap:
                        best_val_ap = val_ap
                        best_config = (hidden_dim, lr, dropout)

    print("\nBest config:", best_config)
    args.hidden_dim, args.lr, args.dropout = best_config

    run_link_prediction(args, train_data, val_data, test_data, return_val_only=False)
    
