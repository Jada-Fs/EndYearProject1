"""Tests for the full production pipeline on demo data."""

from smart_agri_gnn.config import PipelineConfig
from smart_agri_gnn.evaluate import save_prediction_scatter
from smart_agri_gnn.pipeline import run_pipeline
import numpy as np


def test_run_pipeline_on_demo_data(tmp_path):
    config = PipelineConfig(
        data_root=tmp_path,
        use_demo_data=True,
        n_estimators=10,
        n_neighbors=4,
        random_state=42,
    )
    result = run_pipeline(config, train_graph_model=False)
    assert result.processed.target_column == "yield"
    assert result.graph.num_nodes == 20
    assert (tmp_path / "data" / "processed" / "clean_data.parquet").is_file()
    assert (tmp_path / "data" / "graph" / "graph.npz").is_file()
    assert (tmp_path / "models" / "baseline_random_forest.joblib").is_file()
    assert result.baseline_metrics
    assert result.gnn_metrics is None


def test_prediction_scatter(tmp_path):
    path = save_prediction_scatter(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 1.9, 2.8]),
        tmp_path / "pred.png",
        "demo",
    )
    assert path.is_file()
