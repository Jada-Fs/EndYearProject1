"""Tests for k-NN graph construction."""

import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

from smart_agri_gnn.graph import build_knn_edges, build_spatial_graph
from smart_agri_gnn.preprocess import ProcessedData


def test_knn_edges_are_undirected_and_exclude_self():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    edges = build_knn_edges(coords, n_neighbors=2)
    assert edges.shape[0] == 2
    assert not np.any(edges[0] == edges[1])
    pairs = {tuple(sorted(pair)) for pair in edges.T}
    assert (0, 1) in pairs or (0, 2) in pairs


def test_build_spatial_graph_matches_region_rows():
    soil = gpd.GeoDataFrame(
        {
            "country_code": ["UGA", "UGA", "UGA"],
            "region": ["UGA-00", "UGA-01", "UGA-02"],
        },
        geometry=[Point(32.4, 0.3), Point(32.5, 0.3), Point(32.4, 0.4)],
        crs="EPSG:4326",
    )
    frame = pd.DataFrame(
        {
            "region": ["UGA-00", "UGA-01", "UGA-02"],
            "temperature": [0.0, 0.1, -0.1],
            "soil_ph": [0.2, 0.0, -0.2],
            "yield": [2.1, 2.4, 2.5],
        }
    )
    processed = ProcessedData(
        frame=frame,
        join_key="region",
        target_column="yield",
        feature_columns=["temperature", "soil_ph"],
    )
    graph = build_spatial_graph(processed, soil, n_neighbors=3)
    assert graph.num_nodes == 3
    assert graph.num_features == 2
    assert graph.edge_index.shape[1] >= 6
