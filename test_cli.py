from smart_agri_gnn.cli import main
from smart_agri_gnn.gnn import TorchUnavailableError, train_gnn
from smart_agri_gnn.graph import SpatialGraph
import numpy as np
import pytest


def test_cli_demo(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["--demo", "--data-root", str(tmp_path), "--n-estimators", "8"])
    assert code == 0
    captured = capsys.readouterr().out
    assert '"target_column": "yield"' in captured


def test_train_gnn_raises_when_torch_missing(monkeypatch):
    graph = SpatialGraph(
        node_features=np.ones((3, 2), dtype=np.float32),
        targets=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 0]]),
        feature_names=["a", "b"],
        target_column="yield",
    )

    def boom():
        raise TorchUnavailableError("missing")

    monkeypatch.setattr("smart_agri_gnn.gnn._import_torch", boom)
    with pytest.raises(TorchUnavailableError):
        from smart_agri_gnn.config import PipelineConfig

        train_gnn(graph, PipelineConfig())
