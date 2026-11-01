"""Tests for join, impute, and z-score preprocessing."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from smart_agri_gnn.preprocess import process_datasets
from smart_agri_gnn.validation import ValidationError


def _soil(n=4):
    return gpd.GeoDataFrame(
        {
            "country_code": ["UGA"] * n,
            "region": [f"UGA-{i:02d}" for i in range(n)],
            "soil_ph": [6.0, 6.2, 6.4, 6.6][:n],
            "organic_matter": [3.0, 3.1, None, 3.3][:n],
        },
        geometry=[Point(32.4 + i * 0.01, 0.3) for i in range(n)],
        crs="EPSG:4326",
    )


def test_process_aligns_regions_and_scales_features():
    climate = pd.DataFrame(
        {"country_code": ["UGA"], "country": ["Uganda"], "year": [2023], "temperature": [22.4]}
    )
    yield_df = pd.DataFrame({"country_code": ["UGA"] * 4, "yield": [2.1, 2.4, 2.6, 2.8]})
    processed = process_datasets(climate, yield_df, _soil(4))
    assert processed.join_key == "region"
    assert processed.target_column == "yield"
    assert len(processed.frame) == 4
    assert processed.frame["organic_matter"].isna().sum() == 0
    temperature = processed.frame["temperature"]
    assert abs(temperature.mean()) < 1e-8 or temperature.nunique() == 1


def test_process_rejects_unalignable_yield():
    climate = pd.DataFrame(
        {"country_code": ["UGA"], "country": ["Uganda"], "year": [2023], "temperature": [22.4]}
    )
    yield_df = pd.DataFrame({"country_code": ["UGA"] * 2, "yield": [2.1, 2.4]})
    with pytest.raises(ValidationError, match="aligned"):
        process_datasets(climate, yield_df, _soil(4))
