"""
train_maize_gnn.py
==================
Trains a Graph Neural Network (GraphSAGE with message passing) on the dedicated
Uganda Maize dataset (30 districts across 10 Agro-Ecological Zones) for:

1. Maize Yield Forecasting (Continuous Regression: tonnes / hectare)
2. Maize Gray Leaf Spot (GLS) Disease Surveillance (Binary Outbreak Classification)

Compares GraphSAGE against an isolated Random Forest baseline to demonstrate
how spatial message-passing across neighbouring farms/districts improves predictions.
"""

import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainMaizeGNN")

ROOT = Path(__file__).parent.resolve()
GRAPH_PATH = ROOT / "datasets" / "graph" / "graph_maize.pt"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MaizeYieldSAGERegressor(torch.nn.Module):
    """2-Layer Residual GraphSAGE for Continuous Maize Yield Prediction."""
    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.lin_skip = torch.nn.Linear(in_channels, hidden_channels)
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(16, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        # Skip connection
        skip = self.lin_skip(x)
        h1 = self.norm(h1 + skip)
        # Layer 2
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = self.norm(h2 + h1)
        # Output Head
        out = self.head(h2)
        return out.squeeze(-1)


class MaizeDiseaseSAGEClassifier(torch.nn.Module):
    """2-Layer GraphSAGE for Maize Gray Leaf Spot (GLS) Outbreak Surveillance."""
    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(16, 2)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = self.norm(h2)
        logits = self.head(h2)
        return logits


def train_maize_models():
    logger.info("Loading Maize PyG spatial graph from %s", GRAPH_PATH)
    data: Data = torch.load(GRAPH_PATH, weights_only=False)

    X_np = data.x.numpy()
    y_yield_np = data.y_yield.numpy()
    y_disease_np = data.y_disease.numpy()

    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    # ===========================================================================
    # 1. BASELINE: Random Forest (No Graph Message-Passing)
    # ===========================================================================
    logger.info("Training Random Forest Baselines on Maize...")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_np[train_mask], y_yield_np[train_mask])
    rf_pred_yield = rf_reg.predict(X_np[test_mask])

    rf_mae = mean_absolute_error(y_yield_np[test_mask], rf_pred_yield)
    rf_rmse = np.sqrt(mean_squared_error(y_yield_np[test_mask], rf_pred_yield))
    rf_r2 = r2_score(y_yield_np[test_mask], rf_pred_yield)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_np[train_mask], y_disease_np[train_mask])
    rf_pred_disease = rf_clf.predict(X_np[test_mask])

    rf_acc = accuracy_score(y_disease_np[test_mask], rf_pred_disease)
    rf_f1 = f1_score(y_disease_np[test_mask], rf_pred_disease, average="macro", zero_division=0)

    # ===========================================================================
    # 2. GNN: GraphSAGE with Spatial Message Passing
    # ===========================================================================
    logger.info("Training GraphSAGE Regressor (Maize Yield Forecasting)...")
    in_features = data.x.shape[1]
    sage_reg = MaizeYieldSAGERegressor(in_features, hidden_channels=32, dropout=0.2)
    optimizer_reg = torch.optim.Adam(sage_reg.parameters(), lr=0.01, weight_decay=1e-4)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(250):
        sage_reg.train()
        optimizer_reg.zero_grad()
        pred = sage_reg(data.x, data.edge_index)
        loss = F.mse_loss(pred[data.train_mask], data.y_yield[data.train_mask])
        loss.backward()
        optimizer_reg.step()

        sage_reg.eval()
        with torch.no_grad():
            val_pred = sage_reg(data.x, data.edge_index)
            val_loss = F.mse_loss(val_pred[data.val_mask], data.y_yield[data.val_mask]).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = sage_reg.state_dict().copy()

    sage_reg.load_state_dict(best_weights)
    sage_reg.eval()
    with torch.no_grad():
        gnn_pred_yield = sage_reg(data.x, data.edge_index)[data.test_mask].numpy()

    gnn_mae = mean_absolute_error(y_yield_np[test_mask], gnn_pred_yield)
    gnn_rmse = np.sqrt(mean_squared_error(y_yield_np[test_mask], gnn_pred_yield))
    gnn_r2 = r2_score(y_yield_np[test_mask], gnn_pred_yield)

    logger.info("Training GraphSAGE Classifier (Maize Disease Surveillance)...")
    sage_clf = MaizeDiseaseSAGEClassifier(in_features, hidden_channels=32, dropout=0.2)
    optimizer_clf = torch.optim.Adam(sage_clf.parameters(), lr=0.015, weight_decay=1e-4)

    best_val_acc = 0.0
    best_clf_weights = None

    for epoch in range(200):
        sage_clf.train()
        optimizer_clf.zero_grad()
        logits = sage_clf(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y_disease[data.train_mask])
        loss.backward()
        optimizer_clf.step()

        sage_clf.eval()
        with torch.no_grad():
            val_logits = sage_clf(data.x, data.edge_index)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds[data.val_mask] == data.y_disease[data.val_mask]).float().mean().item()
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_clf_weights = sage_clf.state_dict().copy()

    sage_clf.load_state_dict(best_clf_weights)
    sage_clf.eval()
    with torch.no_grad():
        test_logits = sage_clf(data.x, data.edge_index)[data.test_mask]
        gnn_pred_disease = test_logits.argmax(dim=-1).numpy()

    gnn_acc = accuracy_score(y_disease_np[test_mask], gnn_pred_disease)
    gnn_f1 = f1_score(y_disease_np[test_mask], gnn_pred_disease, average="macro", zero_division=0)

    # ===========================================================================
    # 3. COMPARISON & RESULTS SUMMARY
    # ===========================================================================
    yield_improvement = ((rf_mae - gnn_mae) / rf_mae) * 100.0

    print("\n" + "=" * 70)
    print("  UGANDA MAIZE GNN PREDICTION BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Metric':<28} | {'Random Forest':<15} | {'GraphSAGE (GNN)':<15}")
    print("-" * 70)
    print(f"{'Maize Yield MAE (t/ha)':<28} | {rf_mae:<15.4f} | {gnn_mae:<15.4f}  (Improvement: {yield_improvement:+.1f}%)")
    print(f"{'Maize Yield RMSE (t/ha)':<28} | {rf_rmse:<15.4f} | {gnn_rmse:<15.4f}")
    print(f"{'Maize Yield R²':<28} | {rf_r2:<15.4f} | {gnn_r2:<15.4f}")
    print("-" * 70)
    print(f"{'GLS Disease Risk Accuracy':<28} | {rf_acc * 100:<14.1f}% | {gnn_acc * 100:<14.1f}%")
    print(f"{'GLS Disease Risk Macro F1':<28} | {rf_f1:<15.4f} | {gnn_f1:<15.4f}")
    print("=" * 70)

    results_payload = {
        "dataset": "Uganda Maize GNN Dataset",
        "yield_forecasting": {
            "random_forest": {"mae": float(rf_mae), "rmse": float(rf_rmse), "r2": float(rf_r2)},
            "graphsage": {"mae": float(gnn_mae), "rmse": float(gnn_rmse), "r2": float(gnn_r2)},
            "mae_improvement_percent": float(yield_improvement),
        },
        "disease_surveillance": {
            "random_forest": {"accuracy": float(rf_acc), "f1": float(rf_f1)},
            "graphsage": {"accuracy": float(gnn_acc), "f1": float(gnn_f1)},
        }
    }

    out_file = OUTPUT_DIR / "maize_gnn_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    logger.info("Saved quantitative experiment benchmark to %s", out_file)

    # ===========================================================================
    # 4. SAVE MODEL CHECKPOINTS TO models/
    # ===========================================================================
    import joblib
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(sage_reg.state_dict(), MODELS_DIR / "gnn_maize_yield.pt")
    torch.save(sage_clf.state_dict(), MODELS_DIR / "gnn_maize_disease.pt")
    joblib.dump(rf_reg, MODELS_DIR / "rf_maize_yield.joblib")
    joblib.dump(rf_clf, MODELS_DIR / "rf_maize_disease.joblib")

    logger.info("Saved Maize GNN and RF checkpoints to %s", MODELS_DIR)



if __name__ == "__main__":
    train_maize_models()
