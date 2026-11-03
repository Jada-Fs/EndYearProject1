"""
run_experiments.py
==================
End-to-end experiment runner for the Uganda Smart Agriculture GNN project.

Trains and evaluates all models, saves results, and produces a comparison report.
Run this script once to train all models before launching the Streamlit dashboard.

Usage:
    python run_experiments.py
    python run_experiments.py --no-gnn   (skip GNN if PyTorch unavailable)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

# Local modules
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all_datasets
from preprocessing import (
    preprocess_for_disease,
    preprocess_for_forecasting,
    preprocess_for_suitability,
)
from random_forest_models import (
    train_rf_disease_classifier,
    train_rf_suitability_classifier,
    train_rf_yield_regressor,
    cross_validate_rf,
)
from graph_builder import build_graph_from_dataframe, validate_graph
from evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    classification_comparison_table,
    regression_comparison_table,
)

MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def run_experiments(train_gnn: bool = True) -> dict:
    """
    Run the full experimental pipeline.

    Returns a results dict that is also saved to results/experiment_results.json.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 60)
    LOGGER.info("UGANDA SMART AGRICULTURE GNN EXPERIMENT")
    LOGGER.info("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: Load datasets
    # -----------------------------------------------------------------
    LOGGER.info("\n[1/5] Loading datasets...")
    datasets = load_all_datasets()
    base_df = datasets["base"]
    suitability_df = datasets["suitability"]
    disease_df = datasets["disease"]
    forecasting_df = datasets["forecasting"]
    LOGGER.info("Base: %d districts | Suitability: %d rows | Disease: %d rows | Forecasting: %d rows",
                len(base_df), len(suitability_df), len(disease_df), len(forecasting_df))

    # -----------------------------------------------------------------
    # Step 2: Preprocessing
    # -----------------------------------------------------------------
    LOGGER.info("\n[2/5] Preprocessing...")
    split_s = preprocess_for_suitability(suitability_df)
    split_d = preprocess_for_disease(disease_df)
    split_f = preprocess_for_forecasting(forecasting_df)

    # -----------------------------------------------------------------
    # Step 3: Build graph
    # -----------------------------------------------------------------
    LOGGER.info("\n[3/5] Building spatial graph...")
    from preprocessing import FORECASTING_FEATURES
    graph = build_graph_from_dataframe(
        base_df,
        feature_columns=FORECASTING_FEATURES,
        target_column="crop_yield",
        n_neighbors=5,
    )
    warnings = validate_graph(graph)
    if warnings:
        for w in warnings:
            LOGGER.warning("  %s", w)

    # -----------------------------------------------------------------
    # Step 4: Train Random Forest baselines
    # -----------------------------------------------------------------
    LOGGER.info("\n[4/5] Training Random Forest baselines...")

    # 4a. Suitability
    LOGGER.info("  -> Crop Suitability Classifier")
    rf_suit = train_rf_suitability_classifier(
        split_s.X_train, split_s.y_train,
        split_s.X_test, split_s.y_test,
        split_s.feature_names,
        save_dir=MODELS_DIR,
    )
    # Cross-validation
    suit_cv = cross_validate_rf(
        np.vstack([split_s.X_train, split_s.X_test]),
        np.concatenate([split_s.y_train, split_s.y_test]),
        task="classification",
    )
    LOGGER.info("  CV F1: %.3f +/- %.3f", suit_cv["mean"], suit_cv["std"])

    # 4b. Disease risk
    LOGGER.info("  -> Disease Risk Classifier")
    rf_disease = train_rf_disease_classifier(
        split_d.X_train, split_d.y_train,
        split_d.X_test, split_d.y_test,
        split_d.feature_names,
        save_dir=MODELS_DIR,
    )

    # 4c. Yield forecasting
    LOGGER.info("  -> Yield Regression")
    rf_yield = train_rf_yield_regressor(
        split_f.X_train, split_f.y_train,
        split_f.X_test, split_f.y_test,
        split_f.feature_names,
        save_dir=MODELS_DIR,
    )
    yield_cv = cross_validate_rf(
        np.vstack([split_f.X_train, split_f.X_test]),
        np.concatenate([split_f.y_train, split_f.y_test]),
        task="regression",
    )
    LOGGER.info("  CV RMSE: %.4f +/- %.4f", yield_cv["mean"], yield_cv["std"])

    # Save scalers for dashboard use
    joblib.dump(split_s.scaler, MODELS_DIR / "scaler_suitability.joblib")
    joblib.dump(split_d.scaler, MODELS_DIR / "scaler_disease.joblib")
    joblib.dump(split_f.scaler, MODELS_DIR / "scaler_yield.joblib")
    LOGGER.info("  Scalers saved.")

    # -----------------------------------------------------------------
    # Step 5: Train GNN (optional)
    # -----------------------------------------------------------------
    gnn_suit_result = None
    gnn_disease_result = None
    gnn_yield_result = None

    if train_gnn:
        LOGGER.info("\n[5/5] Training GNN models...")
        try:
            from gnn_model import train_gnn as _train_gnn, TorchUnavailableError

            # Build district-level graphs for each task
            suit_base = suitability_df.groupby("district").agg({
                c: "mean" for c in split_s.feature_names + ["suitable"]
                if c in suitability_df.columns
            }).reset_index()
            # Binary target: high versatile suitability (>= median) vs moderate
            suit_base["suitable"] = (suit_base["suitable"] >= suit_base["suitable"].median()).astype(int)

            # GNN for suitability (district level)
            LOGGER.info("  -> GNN Suitability Classifier")
            suit_graph = build_graph_from_dataframe(
                suit_base, feature_columns=split_s.feature_names,
                target_column="suitable", n_neighbors=4,
            )
            gnn_suit_result = _train_gnn(
                suit_graph.node_features,
                suit_graph.targets.astype(int),
                suit_graph.edge_index,
                task="classification",
                num_classes=2,
                hidden_channels=32,
                epochs=250,
                lr=0.015,
                patience=30,
                save_path=MODELS_DIR / "gnn_suitability.pt",
            )
            LOGGER.info("  GNN Suit best epoch: %d | val acc: %.3f",
                        gnn_suit_result.best_epoch, gnn_suit_result.val_metric)

            # GNN for disease risk
            LOGGER.info("  -> GNN Disease Risk Classifier")
            disease_graph = build_graph_from_dataframe(
                disease_df, feature_columns=split_d.feature_names,
                target_column="disease_risk", n_neighbors=4,
            )
            dis_test_mask = np.zeros(len(disease_df), dtype=bool)
            dis_test_mask[split_d.test_indices] = True
            gnn_disease_result = _train_gnn(
                disease_graph.node_features,
                disease_graph.targets.astype(int),
                disease_graph.edge_index,
                task="classification",
                num_classes=2,
                hidden_channels=32,
                dropout=0.0,
                epochs=250,
                lr=0.005,
                min_epochs=15,
                patience=35,
                test_mask=dis_test_mask,
                save_path=MODELS_DIR / "gnn_disease.pt",
            )
            LOGGER.info("  GNN Disease best epoch: %d | val metric: %.3f",
                        gnn_disease_result.best_epoch, gnn_disease_result.val_metric)

            # GNN for yield forecasting
            LOGGER.info("  -> GNN Yield Regressor")
            yield_graph = build_graph_from_dataframe(
                base_df, feature_columns=split_f.feature_names,
                target_column="crop_yield", n_neighbors=5,
            )
            yield_test_mask = np.zeros(len(base_df), dtype=bool)
            yield_test_mask[split_f.test_indices] = True
            gnn_yield_result = _train_gnn(
                yield_graph.node_features,
                yield_graph.targets,
                yield_graph.edge_index,
                task="regression",
                hidden_channels=48,
                dropout=0.1,
                epochs=350,
                lr=0.005,
                min_epochs=15,
                patience=45,
                test_mask=yield_test_mask,
                save_path=MODELS_DIR / "gnn_yield.pt",
            )
            LOGGER.info("  GNN Yield best epoch: %d | val metric: %.4f",
                        gnn_yield_result.best_epoch, gnn_yield_result.val_metric)

        except Exception as exc:
            LOGGER.warning("GNN training failed: %s. Results will show RF only.", exc)

    # -----------------------------------------------------------------
    # Compile results
    # -----------------------------------------------------------------
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("RESULTS SUMMARY")
    LOGGER.info("=" * 60)

    def _metrics_dict(result, task_type):
        if result is None:
            return None
        if task_type == "classification":
            return {
                "accuracy": round(float(compute_classification_metrics(
                    result.true_labels, result.predictions, result.probabilities,
                    "", ""
                ).accuracy), 4),
                "f1": round(float(compute_classification_metrics(
                    result.true_labels, result.predictions, result.probabilities,
                    "", ""
                ).f1), 4),
                "recall": round(float(compute_classification_metrics(
                    result.true_labels, result.predictions, result.probabilities,
                    "", ""
                ).recall), 4),
                "roc_auc": round(float(compute_classification_metrics(
                    result.true_labels, result.predictions, result.probabilities,
                    "", ""
                ).roc_auc or 0), 4),
            }
        else:
            m = compute_regression_metrics(result.true_labels, result.predictions, "", "")
            return {"mae": round(m.mae, 4), "rmse": round(m.rmse, 4), "r2": round(m.r2, 4)}

    results = {
        "suitability": {
            "rf": {
                "accuracy": round(rf_suit.accuracy, 4),
                "precision": round(rf_suit.precision, 4),
                "recall": round(rf_suit.recall, 4),
                "f1": round(rf_suit.f1, 4),
                "roc_auc": round(rf_suit.roc_auc or 0, 4),
                "pr_auc": round(rf_suit.pr_auc or 0, 4),
                "n_train": rf_suit.n_train,
                "n_test": rf_suit.n_test,
                "cv_f1_mean": round(suit_cv["mean"], 4),
                "cv_f1_std": round(suit_cv["std"], 4),
                "feature_importances": rf_suit.feature_importances,
            },
            "gnn": _metrics_dict(gnn_suit_result, "classification") if gnn_suit_result else None,
        },
        "disease": {
            "rf": {
                "accuracy": round(rf_disease.accuracy, 4),
                "precision": round(rf_disease.precision, 4),
                "recall": round(rf_disease.recall, 4),
                "f1": round(rf_disease.f1, 4),
                "roc_auc": round(rf_disease.roc_auc or 0, 4),
                "pr_auc": round(rf_disease.pr_auc or 0, 4),
                "n_train": rf_disease.n_train,
                "n_test": rf_disease.n_test,
                "feature_importances": rf_disease.feature_importances,
            },
            "gnn": _metrics_dict(gnn_disease_result, "classification") if gnn_disease_result else None,
        },
        "yield": {
            "rf": {
                "mae": round(rf_yield.mae, 4),
                "rmse": round(rf_yield.rmse, 4),
                "r2": round(rf_yield.r2, 4),
                "n_train": rf_yield.n_train,
                "n_test": rf_yield.n_test,
                "cv_rmse_mean": round(yield_cv["mean"], 4),
                "cv_rmse_std": round(yield_cv["std"], 4),
                "feature_importances": rf_yield.feature_importances,
            },
            "gnn": _metrics_dict(gnn_yield_result, "regression") if gnn_yield_result else None,
        },
        "graph": graph.summary(),
    }

    # Print comparison
    print("\n--- Crop Suitability ---")
    print(f"  RF  -> Acc={results['suitability']['rf']['accuracy']} | F1={results['suitability']['rf']['f1']}")
    if results["suitability"]["gnn"]:
        print(f"  GNN -> Acc={results['suitability']['gnn']['accuracy']} | F1={results['suitability']['gnn']['f1']}")

    print("\n--- Disease Risk ---")
    print(f"  RF  -> Acc={results['disease']['rf']['accuracy']} | Recall={results['disease']['rf']['recall']}")
    if results["disease"]["gnn"]:
        print(f"  GNN -> Acc={results['disease']['gnn']['accuracy']} | Recall={results['disease']['gnn']['recall']}")

    print("\n--- Yield Forecasting ---")
    print(f"  RF  -> MAE={results['yield']['rf']['mae']} | RMSE={results['yield']['rf']['rmse']} | R2={results['yield']['rf']['r2']}")
    if results["yield"]["gnn"]:
        print(f"  GNN -> MAE={results['yield']['gnn']['mae']} | RMSE={results['yield']['gnn']['rmse']} | R2={results['yield']['gnn']['r2']}")

    # Save results
    results_path = RESULTS_DIR / "experiment_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    LOGGER.info("\nResults saved to %s", results_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Uganda Smart Agriculture experiments")
    parser.add_argument("--no-gnn", action="store_true", help="Skip GNN training")
    args = parser.parse_args()
    run_experiments(train_gnn=not args.no_gnn)
